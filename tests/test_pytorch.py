import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

torch=pytest.importorskip('torch')
from polarized_csi.model_pt import DL_M1
from polarized_csi.dataset_pt import StokesPairDataset
from scripts.smoke_test import generate_data


def run(*args, check=True):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    return subprocess.run([sys.executable,*map(str,args)],env=env,check=check,capture_output=True,text=True)


def test_model_is_affine_and_796_parameters():
    model=DL_M1().eval()
    assert sum(p.numel() for p in model.parameters())==796
    x,y=torch.randn(1,1,8,8),torch.randn(1,1,8,8)
    torch.testing.assert_close(model(.3*x+.7*y),.3*model(x)+.7*model(y))


def test_real_training_validation_checkpoint_and_inference(tmp_path):
    data=generate_data(tmp_path/'data')
    output=tmp_path/'nested/model.pt'
    args=['-m','polarized_csi.train_pt','--train-compressed',data/'train/c','--train-stokes',data/'train/s',
          '--val-compressed',data/'val/c','--val-stokes',data/'val/s','--epochs','2','--batch-size','2','--out',output]
    run(*args)
    record=json.loads(Path(str(output)+'.json').read_text())
    losses=[row['val_mse'] for row in record['history']]
    assert record['selected_epoch']==int(np.argmin(losses))+1
    model=DL_M1()
    model.load_state_dict(torch.load(output,weights_only=True))
    model.eval()
    dataset=StokesPairDataset(data/'val/c',data/'val/s')
    with torch.no_grad():
        actual=np.mean([torch.mean((model(dataset[i][0][None])[0]-dataset[i][1])**2).item() for i in range(len(dataset))])
    assert actual==pytest.approx(min(losses),rel=1e-5)
    pred=tmp_path/'nested/results/pred.npy'
    run('-m','polarized_csi.infer_pt','--weights',output,'--input',data/'val/c/val00.npy','--out-npy',pred)
    assert np.load(pred).shape==(16,16,4)
    assert json.loads(Path(str(pred)+'.json').read_text())['weights']['sha256']
    run('evaluate.py','--pred',pred,'--true',data/'val/s/val00.npy','--out-json',tmp_path/'evaluation.json')
    # Same seed and CPU input reproduce the checkpoint in this environment.
    output2=tmp_path/'repeat.pt'
    run(*args[:-1], output2)
    state2=torch.load(output2,weights_only=True)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,state2[name],rtol=0,atol=0)


def test_invalid_training_does_not_publish_checkpoint(tmp_path):
    data=generate_data(tmp_path/'data')
    np.save(data/'train/s/train00.npy',np.full((16,16,4),np.nan))
    output=tmp_path/'bad.pt'
    result=run('-m','polarized_csi.train_pt','--train-compressed',data/'train/c','--train-stokes',data/'train/s',
               '--epochs','1','--out',output,check=False)
    assert result.returncode!=0
    assert not output.exists()


def test_partial_validation_args_fail(tmp_path):
    result=run('-m','polarized_csi.train_pt','--train-compressed',tmp_path,'--train-stokes',tmp_path,
               '--val-compressed',tmp_path,check=False)
    assert result.returncode!=0
    assert 'both validation' in result.stderr
