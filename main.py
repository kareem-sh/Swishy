from modes.live_stream import run_live_stream
from modes.image_mode import run_image_mode
from modes.video_mode import run_video_mode


# ==================================
# SELECT MODE
# ==================================

MODE = "live"

# MODE = "image"
# MODE = "video"


# ==================================
# RUN
# ==================================

if MODE == "live":

    run_live_stream()

elif MODE == "image":

    run_image_mode(
        "assets/test.jpg"
    )

elif MODE == "video":

    run_video_mode(
        "assets/test.mp4"
    )

else:

    print("Invalid mode selected.")
