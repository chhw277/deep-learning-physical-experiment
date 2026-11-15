from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from compute_stokes import ANGLE_SUFFIXES, compute_stokes
from evaluate import evaluate
from polarized_csi.audit import audit_pairs
from polarized_csi.io import load_measurement, load_stokes, stokes_physics
from polarized_csi.metrics import mean_ssim, psnr_channels
from polarized_csi.paths import check_disjoint, list_pairs


def test_16_bit_intensity_preserved(tmp_path):
    values = np.array([[0, 128, 255, 256, 1000, 32768, 65535]], dtype=np.uint16)
    for ext in ('tif', 'png'):
        path = tmp_path / ('sensor.' + ext)
        Image.fromarray(values).save(path)
        np.testing.assert_allclose(load_measurement(path), values.astype(np.float32)/65535)


def test_image_npy_same_scale(tmp_path):
    gray = np.array([[0, 127, 255]], dtype=np.uint8)
    Image.fromarray(gray).save(tmp_path/'g.png')
    np.save(tmp_path/'g.npy', gray.astype(np.float32)/255)
    np.testing.assert_allclose(load_measurement(tmp_path/'g.png'), load_measurement(tmp_path/'g.npy'))


def test_custom_white_level(tmp_path):
    Image.fromarray(np.array([[0, 2048, 4095]], dtype=np.uint16)).save(tmp_path/'raw.tif')
    np.testing.assert_allclose(load_measurement(tmp_path/'raw.tif',4095), [[0,2048/4095,1]])
    with pytest.raises(ValueError):
        load_measurement(tmp_path/'raw.tif', 1023)


@pytest.mark.parametrize('data', [np.array([[np.nan]]), np.array([[np.inf]]), np.array([[256.]]), np.zeros((2,2,3)), np.array([[-1.]])])
def test_reject_bad_measurements(tmp_path, data):
    np.save(tmp_path/'bad.npy', data)
    with pytest.raises(ValueError):
        load_measurement(tmp_path/'bad.npy')


def test_reject_bad_label_shape_and_nan(tmp_path):
    for data in [np.zeros((4,4,1,4)), np.full((4,4,4), np.nan), np.zeros((2,2,4))]:
        np.save(tmp_path/'bad.npy', data)
        with pytest.raises(ValueError):
            load_stokes(tmp_path/'bad.npy', (4,4))


def test_five_angle_known_state_and_case_insensitive_extension(tmp_path):
    state = np.array([1., .2, -.3, .4])
    for suffix, degrees in zip(ANGLE_SUFFIXES, [0,22.5,45,67.5,90]):
        t = np.deg2rad(degrees)
        intensity = .5*(state[0]+.5*np.sin(4*t)*state[1]+np.sin(2*t)**2*state[2]+np.cos(2*t)*state[3])
        # Floating TIFF avoids quantization in this algebra-only test.
        Image.fromarray(np.full((8,8), intensity, dtype=np.float32)).save(tmp_path/('x'+suffix+'.TIFF'))
    np.testing.assert_allclose(compute_stokes(tmp_path,'x'), np.broadcast_to(state,(8,8,4)), atol=2e-7)
    assert stokes_physics(compute_stokes(tmp_path,'x'))['invalid_pixels'] == 0
    np.save(tmp_path/'x_0.npy', np.zeros((8,8)))
    with pytest.raises(ValueError, match='Ambiguous'):
        compute_stokes(tmp_path,'x')


def test_angle_shape_mismatch(tmp_path):
    for i,suffix in enumerate(ANGLE_SUFFIXES):
        np.save(tmp_path/('x'+suffix+'.npy'),np.zeros((1,8) if i==0 else (8,8)))
    with pytest.raises(ValueError, match='different shapes'):
        compute_stokes(tmp_path,'x')


def test_batch_missing_angle_nonzero_exit(tmp_path):
    np.save(tmp_path/'x_0.npy', np.ones((8,8))*.5)
    run = subprocess.run([sys.executable,'compute_stokes.py','--batch','--raw-dir',str(tmp_path),
                          '--out-dir',str(tmp_path/'out')],capture_output=True,text=True)
    assert run.returncode != 0
    assert not (tmp_path/'out/x.npy').exists()


def test_pairing_and_split_reject_silent_errors(tmp_path):
    c,s=tmp_path/'c',tmp_path/'s'
    c.mkdir(); s.mkdir()
    Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(c/'x.png')
    with pytest.raises(ValueError, match='Unpaired'):
        list_pairs(c,s)
    np.save(s/'x.npy',np.zeros((8,8,4)))
    pairs=list_pairs(c,s)
    with pytest.raises(ValueError, match='overlap'):
        check_disjoint(pairs,pairs)
    np.save(c/'x.npy',np.zeros((8,8)))
    with pytest.raises(ValueError, match='Duplicate'):
        list_pairs(c,s)


def test_demo_is_explicitly_nonphysical():
    root=Path(__file__).resolve().parents[1]/'data/demo/train'
    report=audit_pairs(list_pairs(root/'c', root/'s'))
    assert report['sample_count']==8
    assert report['invalid_pixels']==8929
    assert report['pixels']==10240


def test_metrics_batch_uint8_and_nonconstant_error():
    gt=np.zeros((2,8,8,4),dtype=np.uint8)
    pred=np.ones_like(gt)*10
    np.testing.assert_allclose(psnr_channels(pred,gt,255), 10*np.log10(255**2/100))
    assert mean_ssim(gt,gt)==pytest.approx(1)
    assert mean_ssim(pred,gt,255)<1
    result=evaluate(np.full((8,8,4),.1),np.zeros((8,8,4)),'1.0')
    assert result['mean_psnr_db']==pytest.approx(20)
    assert result['mean_ssim']<1
    identity=evaluate(np.zeros((8,8,4)),np.zeros((8,8,4)))
    assert identity['mean_psnr_db']==99
    assert identity['mean_ssim']==pytest.approx(1)


@pytest.mark.parametrize('data_range',['0','-1','nan','inf'])
def test_bad_metric_range(data_range):
    with pytest.raises(ValueError):
        evaluate(np.zeros((8,8,4)),np.zeros((8,8,4)),data_range)
