import guiff_core.core as core

# core.convert_to_webm("./data/label_targets/", "./data/videos")
core.convert_to_webm(
    "./data/label_targets/",
    "./data/videos",
    {
        "crf": 15,
        "vsync": "cfr",
        "b:v": "1.5M",
        "pix_fmt": "yuv420p",
        "r": 25
    },
)
