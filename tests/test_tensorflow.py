import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

tf=pytest.importorskip('tensorflow')
from polarized_csi.dataset import _load_gray_float
from polarized_csi.infer import load_compressed
from polarized_csi import train
from polarized_csi.model import build_dl_m1, compile_for_stokes_regression


def test_tensorflow_no_double_normalization(tmp_path):
    gray=np.array([[0,128,255]],dtype=np.uint8)
    Image.fromarray(gray).save(tmp_path/'x.png')
    expected=gray.astype(np.float32)/255
    np.testing.assert_allclose(_load_gray_float(tmp_path/'x.png'),expected)
    np.testing.assert_allclose(load_compressed(tmp_path/'x.png')[0,...,0],expected)


def test_actual_training_and_model_roundtrip(tmp_path):
    tf.keras.utils.set_random_seed(42)
    model=build_dl_m1()
    compile_for_stokes_regression(model, learning_rate=.001)
    x=np.full((2,8,8,1),.3,dtype=np.float32)
    y=np.broadcast_to(np.array([.5,.1,-.15,.2],dtype=np.float32),(2,8,8,4)).copy()
    before=[w.numpy().copy() for w in model.trainable_weights]
    loss=model.train_on_batch(x,y)
    assert np.isfinite(np.asarray(loss)).all()
    assert any(not np.array_equal(a,b.numpy()) for a,b in zip(before,model.trainable_weights))
    expected=model(x,training=False).numpy()
    output=tmp_path/'roundtrip.keras'
    model.save(output)
    restored=tf.keras.models.load_model(output,compile=False)
    np.testing.assert_allclose(restored(x,training=False).numpy(),expected,atol=1e-7)


def test_best_checkpoint_not_overwritten(tmp_path):
    """Drive best/worse validation epochs through the actual ModelCheckpoint."""
    model=tf.keras.Sequential([tf.keras.Input((1,)),tf.keras.layers.Dense(1,use_bias=False)])
    output=tmp_path/'nested/best.keras'
    pairs=[]
    for split in ['train','val']:
        c,s=tmp_path/split/'c',tmp_path/split/'s'
        c.mkdir(parents=True); s.mkdir()
        Image.fromarray(np.zeros((8,8),dtype=np.uint8)).save(c/'x.png')
        np.save(s/'x.npy',np.zeros((8,8,4)))
    def fake_fit(*args,callbacks,**kwargs):
        for cb in callbacks:
            cb.set_model(model)
            cb.on_train_begin()
        for epoch,(weight,loss) in enumerate([(1.,.1),(2.,.5)]):
            model.layers[0].set_weights([np.array([[weight]],dtype=np.float32)])
            for cb in callbacks:
                cb.on_epoch_end(epoch,{'val_loss':loss,'loss':loss})
        return type('History',(),{'history':{'loss':[.1,.5],'val_loss':[.1,.5]}})()
    argv=['train','--train-compressed',str(tmp_path/'train/c'),'--train-stokes',str(tmp_path/'train/s'),
          '--val-compressed',str(tmp_path/'val/c'),'--val-stokes',str(tmp_path/'val/s'),'--out',str(output)]
    with patch('sys.argv',argv),patch.object(train,'build_dl_m1',return_value=model),patch.object(model,'fit',side_effect=fake_fit):
        train.main()
    saved=tf.keras.models.load_model(output,compile=False)
    np.testing.assert_array_equal(saved.layers[0].get_weights()[0],[[1.]])
    assert json.loads(Path(str(output)+'.json').read_text())['selected_epoch']==1
