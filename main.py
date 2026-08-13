from modes.live_stream import run_live_stream
from modes.image_mode import run_image_mode
from modes.video_mode import run_video_mode


# ==================================
# SELECT MODE
# ==================================

# MODE = "live"
# MODE = "live"

# MODE = "image"
MODE = "video"   # uses assets/videos/video_07_side_jump_shot.mp4


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

    result = run_video_mode("assets/videos/one_score_one_miss.mp4")
    # print(result)
else:

    print("Invalid mode selected.")
