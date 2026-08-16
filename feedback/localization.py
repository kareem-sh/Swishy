"""Arabic localization for rule-level coaching feedback.

Rule evaluation stays language-neutral: it decides the outcome and retains
the original English message.  JSON serialization calls this module to add an
Arabic rendering without changing the existing ``message`` field consumed by
older clients.
"""

from __future__ import annotations

# Variants mirror the same direction used by BiomechanicsEngine._message:
# excellent, good_low, good_high, needs_work_low, and needs_work_high.
_AR_RULE_FEEDBACK = {
    "knee_flexion_loading": {
        "excellent": "تحميل الساقين ممتاز، والقوة تبدأ من الأرض كما ينبغي.",
        "good_low": "تحميل الساقين جيد. قلّل عمق النزول قليلًا لتحافظ على صعود سلس.",
        "good_high": "تحميل الساقين جيد. اثنِ ركبتيك أعمق قليلًا لتوليد رفع أكبر.",
        "needs_work_low": "ثني الركبتين عميق أكثر من اللازم ويسبب توقفًا في أسفل الحركة. ارفع وضعك قليلًا وحافظ على انسيابية الحركة.",
        "needs_work_high": "اثنِ ركبتيك أكثر قبل الصعود حتى تدفع الساقان الكرة بدل الاعتماد على الذراع فقط.",
    },
    "hip_hinge_loading": {
        "excellent": "وضع الوركين ممتاز، والصدر مرفوع والوزن متوازن فوق القدمين.",
        "good_low": "وضع الوركين جيد. افتحهما قليلًا ليرتفع صدرك باتجاه السلة.",
        "good_high": "وضع الوركين جيد. ادفعهما للخلف قليلًا لتحميل قوة أكبر.",
        "needs_work_low": "أنت تنحني إلى الأمام من الخصر. أبقِ صدرك مرفوعًا باتجاه السلة أثناء النزول.",
        "needs_work_high": "وضع الوركين قائم أكثر من اللازم لتوليد القوة. ادفعهما للخلف أثناء النزول كأنك تجلس على مقعد مرتفع.",
    },
    "elbow_slot_ball_lift": {
        "excellent": "الكرة في موضع قوي ومناسب لبدء التصويب.",
        "good_low": "موضع الكرة جيد. أبعدها قليلًا عن صدرك قبل رفعها.",
        "good_high": "موضع الكرة جيد. قرّبها قليلًا أثناء الرفع.",
        "needs_work_low": "مرفقك ملاصق للجسم أكثر من اللازم. اترك له مساحة حتى ترتفع الكرة في خط مستقيم.",
        "needs_work_high": "مرفقك متجه إلى الخارج أكثر من اللازم، مما يجعل الكرة تنحرف أثناء الرفع. ضعه تحت الكرة.",
    },
    "guide_hand_support_lift": {
        "excellent": "اليد المساعدة تسند جانب الكرة جيدًا أثناء الرفع.",
        "good_high": "اليد المساعدة تصل إلى الكرة. قرّبها قليلًا من الجانب ليبقى الرفع في المنتصف.",
        "needs_work_high": "اليد المساعدة بعيدة عن الكرة أثناء الرفع. اسند جانب الكرة حتى تبدأ ذراع التصويب مرحلة الإطلاق.",
    },
    "shoulder_alignment_lift": {
        "excellent": "موضع كتف التصويب مناسب خلال رفع الكرة وإطلاقها.",
        "good_low": "موضع الكتف جيد. ارفعه قليلًا أثناء تجهيز الكرة.",
        "good_high": "موضع الكتف جيد. اخفضه قليلًا لتبقى الذراع مرتاحة عند الإطلاق.",
        "needs_work_low": "كتف التصويب منخفض أثناء الرفع. ارفعه حتى تعمل الذراع تحت الكرة لا بجانبها.",
        "needs_work_high": "كتف التصويب مرتفع أكثر من اللازم عند الإطلاق. اخفضه حتى تمتد الذراع بحرية.",
        "measured": "تم قياس ارتفاع الكتف للعرض فقط، ولا يدخل في النتيجة.",
    },
    "jump_elevation": {
        "excellent": "ارتفاع القفزة جيد وتُطلق الكرة قرب قمة القفزة.",
        "good_low": "ارتفاعك جيد. زيادة بسيطة في القفزة تمنحك مساحة أكبر فوق المدافع.",
        "good_high": "ارتفاعك جيد. خفّض قوة القفزة قليلًا لتصبح الحركة أكثر ثباتًا من تسديدة لأخرى.",
        "needs_work_low": "تكاد لا تغادر الأرض. اصعد مع القفزة وأطلق الكرة قرب القمة.",
        "needs_work_high": "تقفز بقوة أكبر مما تحتاجه التسديدة، وهذا يقلل الثبات. خفّض الارتفاع ودع الذراع تكمل الحركة.",
    },
    "trunk_posture": {
        "excellent": "يبقى جذعك مستقيمًا ومواجهًا للسلة عند خروج الكرة.",
        "good_high": "وضع الجذع جيد. أبقِ صدرك موجهًا نحو السلة أثناء خروج الكرة.",
        "needs_work_high": "يميل الجزء العلوي من جسمك عند الإطلاق، مما يسحب التسديدة عن خطها. أبقِ صدرك نحو السلة.",
    },
    "jump_release_timing": {
        "excellent": "تخرج الكرة أثناء الصعود أو عند قمة القفزة، فتنتقل قوة الرفع إلى التسديدة.",
        "good_high": "الإطلاق يأتي بعد قمة القفزة بقليل. أطلق الكرة أبكر قليلًا حتى يستمر الرفع في دعمها.",
        "needs_work_high": "تنتظر حتى يبدأ جسمك بالهبوط قبل إطلاق الكرة. أطلقها أثناء الصعود أو عند القمة لتستفيد من قوة القفزة.",
    },
    "elbow_extension_release": {
        "excellent": "تمتد ذراعك بالكامل باتجاه السلة عند الإطلاق.",
        "good_low": "امتدادك جيد. مدّ الذراع أكثر قليلًا باتجاه السلة لرفع القوس.",
        "good_high": "امتدادك جيد. خفّف نهاية المد قليلًا ليبقى الإطلاق سلسًا.",
        "needs_work_low": "مدّ مرفقك أكثر عند الإطلاق؛ ما زال مثنيًا عند خروج الكرة وهذا يخفض القوس.",
        "needs_work_high": "تفرد مرفقك بقوة زائدة. مدّ الذراع نحو السلة واترك النهاية مرنة.",
    },
    "release_height": {
        "excellent": "نقطة الإطلاق مرتفعة جيدًا، وهذا يحمي التسديدة ويزيد انحدار القوس.",
        "good_low": "نقطة الإطلاق جيدة. أطلق الكرة أعلى قليلًا لزيادة انحدار القوس.",
        "good_high": "نقطة الإطلاق جيدة ومرتفعة. حافظ عليها من دون مبالغة في المد.",
        "needs_work_low": "تُطلق الكرة مبكرًا وفي موضع منخفض. أطلقها عند قمة الصعود لا في بدايته.",
        "needs_work_high": "تبالغ في المد عند الإطلاق وتفقد التحكم. أطلق الكرة أبكر قليلًا.",
    },
    "follow_through_hold": {
        "excellent": "تحافظ على وضعية المتابعة جيدًا بعد خروج الكرة.",
        "good_low": "المتابعة جيدة. أبقِ اليد مرفوعة باتجاه السلة مدة أطول قليلًا.",
        "needs_work_low": "تنخفض الذراع فور خروج الكرة. أبقِ المرفق واليد مرفوعين نحو السلة حتى تصل الكرة.",
    },
    "landing_balance": {
        "excellent": "تهبط بثبات وتعود ساقاك تحت جسمك.",
        "needs_work_low": "تنخفض كثيرًا بعد الهبوط. امتص الهبوط ثم عد إلى وضع متوازن.",
        "needs_work_high": "تنهي التسديدة من دون الاستقرار على الأرض. اهبط بتحكم واستعد وضعيتك.",
    },
    "wrist_cock_loading": {
        "excellent": "تهيئة الرسغ ممتازة، واليد مثنية للخلف وجاهزة تحت الكرة.",
        "good_low": "تهيئة الرسغ جيدة. قلّل ثني اليد للخلف قليلًا ليبقى الإطلاق سريعًا.",
        "good_high": "تهيئة الرسغ جيدة. اثنِ اليد للخلف قليلًا أكثر لتحميل حركة النقر.",
        "needs_work_low": "الرسغ مثني للخلف أكثر من اللازم أثناء التحميل. خفّف الثني حتى لا يطول مسار الإطلاق.",
        "needs_work_high": "الرسغ مسطح أثناء التحميل. اثنه للخلف تحت الكرة لتوليد نقرة فعالة.",
    },
    "wrist_snap_release": {
        "excellent": "نقرة الرسغ نظيفة، وتنهي اليد حركتها عبر الكرة.",
        "good_low": "نقرة الرسغ موجودة. ادفع اليد عبر الكرة أكثر قليلًا لإكمال الخط.",
        "good_high": "نقرة الرسغ جيدة. أبقها مرتاحة حتى تتدحرج الكرة عن الأصابع بدل دفعها.",
        "needs_work_low": "الرسغ لا ينقر عبر الكرة عند الإطلاق. أنهِ الحركة بدفع اليد إلى الأسفل عبر الكرة.",
        "needs_work_high": "الرسغ يتجاوز امتداده الطبيعي عند الإطلاق. دعه ينهي الحركة بصورة طبيعية.",
    },
    "knee_excursion_loading": {
        "excellent": "حركة الساقين قوية وتمر عبر التسديدة، وهذا هو مصدر القوة.",
        "good_low": "دفع الساقين جيد. اسمح لهما بالنزول أكثر قليلًا حتى يحمل الصعود التسديدة.",
        "needs_work_low": "الانخفاض قبل التسديد محدود جدًا. دع الساقين تنخفضان ثم تصعدان حتى تبدأ القوة من الأرض لا من الذراع.",
    },
    "trajectory_release_angle": {
        "excellent": "خرجت الكرة بزاوية الإطلاق المستهدفة.",
        "good_low": "زاوية الإطلاق منخفضة قليلًا. وجّه الكرة أعلى قليلًا نحو القوس المرجعي.",
        "good_high": "زاوية الإطلاق مرتفعة قليلًا. استخدم مسارًا أكثر تسطحًا نحو القوس المرجعي.",
        "needs_work_low": "زاوية الإطلاق منخفضة أكثر من اللازم. ارفع مسار الكرة نحو القوس المرجعي.",
        "needs_work_high": "زاوية الإطلاق مرتفعة أكثر من اللازم. خفّض المسار مع الحفاظ على القوس المرجعي.",
    },
    "trajectory_release_speed": {
        "excellent": "سرعة إطلاق الكرة مطابقة للسرعة المرجعية.",
        "good_low": "سرعة الكرة منخفضة قليلًا لهذه المسافة. أضف قوة بسيطة باتجاه السلة.",
        "good_high": "سرعة الكرة مرتفعة قليلًا لهذه المسافة. استخدم إطلاقًا أكثر نعومة.",
        "needs_work_low": "سرعة الكرة منخفضة أكثر من اللازم لهذه المسافة. زد القوة مع الحفاظ على القوس المستهدف.",
        "needs_work_high": "سرعة الكرة مرتفعة أكثر من اللازم لهذه المسافة. خفّف الإطلاق مع الحفاظ على القوس المستهدف.",
    },
    "trajectory_apex_height": {
        "excellent": "ارتفاع قمة القوس مطابق للمرجع.",
        "good_low": "القوس منخفض قليلًا. امنح الكرة ارتفاعًا أكبر.",
        "good_high": "القوس مرتفع قليلًا. قرّب القمة من الارتفاع المرجعي.",
        "needs_work_low": "القوس منخفض أكثر من اللازم. ارفع القمة لتحسين دخول الكرة إلى السلة.",
        "needs_work_high": "القوس مرتفع أكثر من اللازم. اخفض القمة مع الحفاظ على دخول متحكم به.",
    },
    "trajectory_apex_progress": {
        "excellent": "بلغت الكرة قمتها في الموضع المستهدف بين الإطلاق والسلة.",
        "good_low": "بلغت الكرة القمة مبكرًا قليلًا. مدّد الصعود أكثر باتجاه السلة.",
        "good_high": "بلغت الكرة القمة متأخرًا قليلًا. اجعلها تبدأ الهبوط أبكر قليلًا.",
        "needs_work_low": "بلغت الكرة القمة مبكرًا أكثر من اللازم. مدّد المسار الصاعد باتجاه السلة.",
        "needs_work_high": "بلغت الكرة القمة متأخرًا أكثر من اللازم. شكّل القوس ليبدأ الهبوط أبكر.",
    },
    "trajectory_apex_distance": {
        "measured": "مسافة قمة القوس معروضة للتفاصيل فقط، ولا تدخل في النتيجة.",
    },
    "guide_wrist_release_proxy": {
        "measured": "محاذاة رسغ اليد المساعدة معروضة كمؤشر تقريبي للدفع فقط، ولا تدخل في النتيجة.",
    },
    "head_stability": {
        "measured": "حركة الرأس والعينين معروضة للمرجع فقط، ولا تدخل في النتيجة.",
    },
    "index_alignment_release": {
        "measured": "اتجاه الأصابع معروض للمرجع فقط، ولا يدخل في النتيجة.",
    },
    "release_height_ratio": {
        "measured": "ارتفاع الإطلاق نسبةً إلى طول اللاعب معروض للمرجع فقط، ولا يدخل في النتيجة.",
    },
    "follow_through_elbow": {
        "measured": "زاوية متابعة المرفق معروضة للمرجع فقط، ولا تدخل في النتيجة.",
    },
    "follow_through_index": {
        "measured": "وضعية نهاية الأصابع معروضة للمرجع فقط، ولا تدخل في النتيجة.",
    },
}

# The incoming offline-peak branch names the scored wrist-snap rule
# ``wrist_release``. Keep the earlier descriptive id as an alias so payloads
# from either side of the merge receive the same translation.
_AR_RULE_FEEDBACK["wrist_release"] = _AR_RULE_FEEDBACK["wrist_snap_release"]


_AR_COACHING_TEXT = {
    "Keep your full body visible to the camera.": (
        "أبقِ جسمك كاملًا ظاهرًا أمام الكاميرا."
    ),
    "Shot was cut off before landing — let the rep finish so follow-through and balance can be scored.": (
        "انقطع تسجيل التسديدة قبل الهبوط. اترك المحاولة تكتمل حتى يمكن تقييم المتابعة والتوازن."
    ),
    "Excellent mechanics — keep this rhythm.": (
        "ميكانيكية التسديدة ممتازة، حافظ على هذا الإيقاع."
    ),
    "Solid form. Refine the notes above.": (
        "الأداء متماسك. حسّن الملاحظات المذكورة أعلاه."
    ),
    "Fair attempt — focus on one correction per rep.": (
        "المحاولة مقبولة. ركّز على تعديل واحد في كل تكرار."
    ),
    "Break the shot into phases and practice each slowly.": (
        "قسّم التسديدة إلى مراحل وتدرّب على كل مرحلة ببطء."
    ),
    "Nothing to change. Keep these reps and start shooting them at game speed.": (
        "لا يوجد ما يحتاج إلى تغيير. حافظ على هذه التكرارات وابدأ التنفيذ بسرعة المباراة."
    ),
    "Take it one piece at a time: load, lift, release, hold the finish.": (
        "تدرّب على جزء واحد كل مرة: التحميل، والرفع، والإطلاق، ثم تثبيت المتابعة."
    ),
    (
        "Your feet left the floor, but not enough to be a jump shot. "
        "Pick one and repeat it: stay planted through the release, or "
        "commit to the jump. Half of each is the hardest to repeat. On a "
        "free throw, staying planted is also the safer choice — leaving "
        "the floor is legal, but landing over the line before the ball "
        "reaches the ring is a violation."
    ): (
        "غادرت قدماك الأرض، لكن ليس بما يكفي لتُعدّ التسديدة قفزية. اختر أسلوبًا واحدًا وكرره: "
        "إما أن تبقى ثابتًا حتى الإطلاق، أو تلتزم بالقفزة كاملة. المزج بينهما هو الأصعب في التكرار. "
        "وفي الرمية الحرة يكون البقاء على الأرض الخيار الأكثر أمانًا؛ فالقفز قانوني، لكن الهبوط بعد "
        "خط الرمية قبل أن تصل الكرة إلى الحلقة يُعد مخالفة."
    ),
}


_AR_DRILLS = {
    "Load drill: 10 reps pausing at the bottom of your knee bend (hips back, chest tall) before rising into the shot.": (
        "تمرين التحميل: 10 تكرارات مع توقف قصير في أسفل ثني الركبتين، والوركان للخلف والصدر مرفوع، قبل الصعود للتسديد."
    ),
    "Hip-hinge drill: practice 10 half-squats keeping shins vertical and hips sitting back without folding at the waist.": (
        "تمرين مفصل الورك: نفّذ 10 أنصاف قرفصاء مع إبقاء الساقين عموديتين ودفع الوركين للخلف من دون الانحناء عند الخصر."
    ),
    "One-hand form shooting from chest: 20 reps keeping elbow under the ball and pointing at the rim through the lift.": (
        "تمرين التصويب بيد واحدة من الصدر: 20 تكرارًا مع إبقاء المرفق تحت الكرة وموجهًا نحو السلة أثناء الرفع."
    ),
    "Guide-hand form shooting: take 15 close-range reps with the guide hand supporting the side during the lift, then coming cleanly off the ball at release.": (
        "تمرين اليد المساعدة: 15 تسديدة قريبة مع إسناد جانب الكرة أثناء الرفع، ثم إبعاد اليد عنها بوضوح عند الإطلاق."
    ),
    "Wall touch drill: lift the ball while keeping shooting shoulder elevated — 15 slow reps without dropping the elbow.": (
        "تمرين لمس الحائط: ارفع الكرة مع إبقاء كتف التصويب مرتفعًا؛ 15 تكرارًا بطيئًا من دون خفض المرفق."
    ),
    "Tall-posture reps: shoot 10 times focusing on staying stacked (ears over hips) through the entire motion.": (
        "تمرين الوضعية المستقيمة: نفّذ 10 تسديدات مع إبقاء الأذنين فوق الوركين طوال الحركة."
    ),
    "Eyes-up drill: pick a rim spot and hold your gaze through release for 10 shots — no head dip or pull-away.": (
        "تمرين تثبيت النظر: اختر نقطة على الحلقة وثبّت نظرك حتى الإطلاق في 10 تسديدات، من دون خفض الرأس أو سحبه للخلف."
    ),
    "Finger-through drill: 20 close-range shots snapping index finger down through the ball at release.": (
        "تمرين مرور الإصبع: 20 تسديدة قريبة مع نقر السبابة إلى الأسفل عبر الكرة عند الإطلاق."
    ),
    "Extension pause: at the top of each shot, freeze with full elbow extension for 1 second before follow-through — 15 reps.": (
        "تمرين تثبيت الامتداد: عند أعلى كل تسديدة، ثبّت المرفق ممدودًا بالكامل لثانية واحدة قبل المتابعة؛ 15 تكرارًا."
    ),
    "Peak-release drill: take 10 controlled jump shots and let the ball leave on the rise or at the top, before your body starts coming down.": (
        "تمرين الإطلاق عند القمة: نفّذ 10 تسديدات قفزية متحكمًا بها، وأطلق الكرة أثناء الصعود أو عند القمة قبل أن يبدأ جسمك بالهبوط."
    ),
    "High-release drill: shoot from chin level only after the ball passes your forehead — 10 slow reps emphasizing height.": (
        "تمرين الإطلاق المرتفع: ابدأ من مستوى الذقن ولا تطلق الكرة إلا بعد أن تتجاوز الجبهة؛ 10 تكرارات بطيئة مع التركيز على الارتفاع."
    ),
    "Hold finish: after every shot, keep elbow locked and arm extended for 2 seconds — 20 reps.": (
        "تمرين تثبيت النهاية: بعد كل تسديدة، أبقِ المرفق ثابتًا والذراع ممدودة لمدة ثانيتين؛ 20 تكرارًا."
    ),
    "Gooseneck hold: finish every rep with index finger pointing at the floor for 2 seconds — 20 close-range reps.": (
        "تمرين تثبيت الرسغ: أنهِ كل تكرار والسبابة تشير إلى الأرض لمدة ثانيتين؛ 20 تكرارًا من مسافة قريبة."
    ),
    "Stick landing: after release, land in the same spot you jumped from — 10 jump shots marking your take-off point.": (
        "تمرين تثبيت الهبوط: بعد الإطلاق، اهبط في الموضع نفسه الذي قفزت منه؛ 10 تسديدات قفزية مع تحديد نقطة الارتقاء."
    ),
    "Game speed: 5 makes from 5 spots, one form check between each spot.": (
        "تمرين سرعة المباراة: سجّل 5 تسديدات من 5 مواقع، مع مراجعة واحدة للأداء بين كل موقع وآخر."
    ),
}


_AR_PHASE_LABELS = {
    "Loading": "التحميل",
    "Ball Lift": "رفع الكرة",
    "Jump": "القفز",
    "Release": "الإطلاق",
    "Follow-Through": "المتابعة",
    "Landing": "الهبوط",
}


_AR_FALLBACK = {
    "excellent": "هذا الجانب ضمن النطاق الممتاز.",
    "good": "هذا الجانب جيد، ويمكن تحسينه قليلًا.",
    "good_low": "هذا الجانب جيد، ويمكن رفع قيمته قليلًا نحو النطاق المثالي.",
    "good_high": "هذا الجانب جيد، ويمكن خفض قيمته قليلًا نحو النطاق المثالي.",
    "needs_work": "هذا الجانب يحتاج إلى تعديل.",
    "needs_work_low": "هذه القيمة أقل من النطاق المطلوب وتحتاج إلى الرفع.",
    "needs_work_high": "هذه القيمة أعلى من النطاق المطلوب وتحتاج إلى الخفض.",
    "measured": "هذا القياس معروض للمرجع فقط، ولا يدخل في النتيجة.",
}


def _variant(rule) -> str:
    """Return the outcome/direction variant used for Arabic feedback."""
    if not rule.scored:
        return "measured"
    outcome = getattr(rule.outcome, "value", str(rule.outcome))
    if outcome == "excellent":
        return "excellent"

    measured = rule.measured_value
    if outcome == "good":
        if measured is not None and rule.ideal_min is not None:
            if measured < rule.ideal_min:
                return "good_low"
        if measured is not None and rule.ideal_max is not None:
            if measured > rule.ideal_max:
                return "good_high"
        return "good"

    if measured is not None and rule.min_value is not None:
        if measured < rule.min_value:
            return "needs_work_low"
    if measured is not None and rule.max_value is not None:
        if measured > rule.max_value:
            return "needs_work_high"
    return "needs_work"


def rule_feedback_ar(rule) -> str:
    """Return Arabic coaching text for a serialized RuleResult."""
    variant = _variant(rule)
    translations = _AR_RULE_FEEDBACK.get(rule.rule_id, {})
    return (
        translations.get(variant)
        or translations.get(getattr(rule.outcome, "value", str(rule.outcome)))
        or translations.get("measured" if not rule.scored else "default")
        or _AR_FALLBACK[variant]
    )


def coaching_text_ar(text: str, rules=()) -> str:
    """Translate one generated coaching/focus/drill string to Arabic.

    Rule messages are translated from the RuleResult that produced them, so
    directional feedback remains aligned with the measured value. The small
    static tables cover score summaries, capture warnings, performance-plan
    notes, and drills. Unknown caller-supplied text is preserved instead of
    being replaced with an invented translation.
    """
    for rule in rules:
        if rule.message == text:
            return rule_feedback_ar(rule)

    translated = _AR_COACHING_TEXT.get(text) or _AR_DRILLS.get(text)
    if translated is not None:
        return translated

    start_prefix = "Recording began mid-shot at "
    if text.startswith(start_prefix):
        label = text[len(start_prefix):].split(" —", 1)[0]
        phase = _AR_PHASE_LABELS.get(label, label)
        return (
            f"بدأ التسجيل في منتصف التسديدة عند مرحلة {phase}. "
            "ابدأ التصوير قبل مرحلة التحميل في المرة القادمة للحصول على تقييم كامل للأداء."
        )

    return text


def localized_coaching(items, rules=()) -> dict:
    """Return parallel English and Arabic arrays for one coaching section."""
    english = list(items)
    return {
        "en": english,
        "ar": [coaching_text_ar(item, rules) for item in english],
    }
