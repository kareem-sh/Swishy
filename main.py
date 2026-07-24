from modes.live_stream import run_live_stream
from modes.image_mode import run_image_mode
from modes.video_mode import run_video_mode
import pipeline


# ==================================
# SELECT MODE
# ==================================

# MODE = "live"
# MODE = "live"

# MODE = "image"
MODE = "video"   # uses assets/test.mp4 (Mixkit jump-shot clip)


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

    result = run_video_mode("assets/videos/MikeDunnBasketballShooting2.mov")
    print(result)
else:

    print("Invalid mode selected.")
