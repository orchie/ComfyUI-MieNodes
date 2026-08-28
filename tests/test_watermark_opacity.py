"""Tests for the `opacity` (透明度) widget added to both image watermark nodes.

AddTextWatermarkForImage|Mie and AddNumberWatermarkForImage|Mie both accept a
new required ``opacity`` widget (FLOAT 0..1, default 1.0).

Both implementations render a fully-opaque watermark first, then blend it
against the original image with a per-pixel weight ``coverage * opacity``.
Because of that, for any opacity ``o`` the output is exactly linear:

    out(o) = base + o * (out(1.0) - base)   (pixel-wise)

The schema test guards the widget contract, the zero/one tests pin the
endpoints, and the linearity tests pin the blend math in between.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_plugin_imports import load_plugin_module  # noqa: E402

import torch  # noqa: E402


WATERMARK_NODES = {
    "AddTextWatermarkForImage|Mie": dict(
        text="MieNodes",
        font_size=24,
        position_x=50.0,
        position_y=50.0,
        color_r=255,
        color_g=255,
        color_b=255,
        outline=False,
        outline_width=0,
        align="center",
    ),
    "AddNumberWatermarkForImage|Mie": dict(
        start_number=1,
        position_x=50.0,
        position_y=50.0,
        font_scale=1.0,
        color_r=255,
        color_g=255,
        color_b=255,
        thickness=2,
        outline=False,
        outline_thickness=1,
    ),
}


@pytest.fixture(scope="module")
def plugin():
    return load_plugin_module()


def make_images(batch=1, h=64, w=128, value=0.0):
    # Black background: uint8 quantization is lossless (0 -> 0), so the
    # linearity assertions below hold without quantization ambiguity.
    return torch.full((batch, h, w, 3), value, dtype=torch.float32)


# ------------------------------ schema ----------------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_widget_schema(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    spec = cls.INPUT_TYPES()
    opacity = spec["required"]["opacity"]
    assert opacity[0] == "FLOAT"
    opts = opacity[1]
    assert opts["default"] == 1.0
    assert opts["min"] == 0.0
    assert opts["max"] == 1.0


# ------------------------------ endpoints -------------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_zero_leaves_image_unchanged(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    out = node.apply_watermark(images, opacity=0.0, **kwargs)[0]
    assert out.shape == images.shape
    assert out.dtype == images.dtype
    assert torch.equal(out, images), "opacity=0 must leave the image untouched"


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_one_renders_full_watermark(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])
    out = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    assert out.shape == images.shape
    assert not torch.equal(out, images), "opacity=1 must actually draw the watermark"
    diff = (out - images).abs()
    assert diff.max().item() > 0.5, "watermark pixels should move toward the text color"


# ----------------------------- blend math -------------------------------- #

@pytest.mark.parametrize("node_name", WATERMARK_NODES)
@pytest.mark.parametrize("opacity", [0.25, 0.5, 0.75])
def test_opacity_blends_linearly(plugin, node_name, opacity):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])

    # Use the node's own opacity=0 output as `base`: both pipelines quantize
    # the image to uint8 internally, so the linear blend is exact relative to
    # that baseline rather than to the raw float input tensor.
    base = node.apply_watermark(images, opacity=0.0, **kwargs)[0]
    full = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    partial = node.apply_watermark(images, opacity=opacity, **kwargs)[0]

    expected = base + opacity * (full - base)
    assert torch.allclose(partial, expected, atol=2.0 / 255.0), (
        f"{node_name}: out({opacity}) must equal base + {opacity}*(out(1) - base)"
    )


@pytest.mark.parametrize("node_name", WATERMARK_NODES)
def test_opacity_clamps_out_of_range(plugin, node_name):
    cls = plugin.NODE_CLASS_MAPPINGS[node_name]
    node = cls()
    images = make_images()
    kwargs = dict(WATERMARK_NODES[node_name])

    over = node.apply_watermark(images, opacity=5.0, **kwargs)[0]
    one = node.apply_watermark(images, opacity=1.0, **kwargs)[0]
    under = node.apply_watermark(images, opacity=-1.0, **kwargs)[0]
    zero = node.apply_watermark(images, opacity=0.0, **kwargs)[0]

    assert torch.equal(over, one), "opacity > 1 must clamp to 1.0"
    assert torch.equal(under, zero), "opacity < 0 must clamp to 0.0"
