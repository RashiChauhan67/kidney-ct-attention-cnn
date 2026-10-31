import numpy as np

from kidney_dacnn.model import NOTEBOOK_DENSE_UNITS, PAPER_FIGURE_DENSE_UNITS, build_model


def test_default_matches_the_saved_notebook():
    # model.summary() in the notebook: 414,389 trainable + 448 non-trainable parameters.
    assert build_model().count_params() == 414_837


def test_paper_figure_head_is_about_the_size_stated_in_the_paper():
    params = build_model(dense_units=PAPER_FIGURE_DENSE_UNITS).count_params()
    assert params == 225_909
    assert abs(params / 1e6 - 0.22) < 0.01  # the paper states 0.22 M parameters


def test_notebook_head_has_the_extra_dense_512_layer():
    assert NOTEBOOK_DENSE_UNITS == (512, 256, 64)
    assert PAPER_FIGURE_DENSE_UNITS == (256, 64)


def test_forward_pass_gives_a_probability_distribution():
    model = build_model(input_shape=(32, 32, 3))
    out = model.predict(np.random.rand(3, 32, 32, 3).astype("float32"), verbose=0)
    assert out.shape == (3, 4)
    np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-5)
