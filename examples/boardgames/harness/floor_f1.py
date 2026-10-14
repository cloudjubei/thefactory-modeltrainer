"""§3.6 F1 — the whole game, built. L2's rule (the first level holding any safe move decides) made strategies five
times smaller and projected the whole game below the trained net on fresh positions (h262); a library of maps found
elsewhere projected it to ~159K bits (h268). F1 builds every one of the opening's 671 frontier positions — in W1's
seeded order, in waves of 32, each wave starting from a library of every map found so far, seeded with the 90 maps L2
found (8,618 bits, paid once) — with L2's rule and pipeline: a certified first-player strategy for the whole game.
Its size is the opening, the seed library once, and every position's own bits (a map is charged in the build that
found it). Judged on completion (all 671; refuted when more than 7 do not finish); on that size against W9's
projection of 271,740 bits (no more supports, more than W9's 95% high of 476,290 refutes, a partial total counting as
a lower bound); and against the trained net's ~660K bits (below supports only when complete)."""
from __future__ import annotations

from harness.floor_w1 import _integrity
from harness.floor_w10 import SPEC as W10

SPEC = {**{k: v for k, v in W10.items() if k not in ("w9_bits", "support", "library_support", "library_refute")},
        "measurement_fp": "4242d4bca6e2",
        "indices": [
            362, 87, 316, 346, 230, 323, 393, 263, 471, 178, 95, 486, 369, 506, 594, 324, 512, 534, 51, 374,
            144, 413, 463, 251, 170, 667, 533, 197, 535, 403, 175, 98, 271, 53, 565, 428, 303, 639, 625, 457,
            615, 180, 504, 384, 135, 301, 669, 308, 49, 120, 638, 70, 564, 499, 631, 327, 236, 528, 104, 576,
            54, 19, 375, 536, 379, 46, 614, 572, 112, 426, 353, 208, 35, 28, 458, 652, 558, 453, 14, 114, 500,
            223, 505, 556, 25, 585, 194, 126, 252, 228, 541, 385, 184, 593, 479, 653, 355, 511, 313, 246, 509,
            288, 543, 416, 258, 597, 354, 467, 244, 272, 596, 561, 649, 261, 147, 75, 189, 333, 109, 74, 513,
            591, 424, 127, 149, 94, 407, 472, 78, 574, 616, 361, 550, 186, 448, 477, 115, 318, 136, 257, 440,
            142, 276, 3, 29, 201, 389, 171, 209, 366, 559, 240, 425, 391, 242, 191, 474, 102, 77, 478, 360,
            442, 443, 214, 485, 96, 218, 515, 363, 217, 55, 629, 377, 309, 654, 141, 322, 497, 274, 248, 378,
            581, 637, 507, 264, 480, 611, 642, 235, 315, 302, 164, 15, 69, 31, 632, 280, 8, 52, 24, 514, 387,
            62, 134, 4, 67, 617, 359, 140, 386, 306, 602, 530, 626, 537, 12, 33, 349, 545, 338, 340, 16, 429,
            233, 122, 163, 579, 123, 66, 193, 23, 566, 590, 358, 275, 553, 332, 398, 130, 344, 627, 557, 548,
            584, 661, 337, 211, 157, 450, 383, 612, 138, 229, 570, 64, 621, 466, 521, 145, 624, 529, 314, 555,
            231, 495, 435, 595, 665, 273, 464, 121, 598, 155, 418, 357, 640, 524, 655, 347, 93, 81, 50, 183,
            554, 650, 10, 11, 538, 89, 348, 606, 644, 630, 430, 370, 481, 179, 294, 101, 436, 146, 350, 73,
            42, 221, 600, 174, 388, 305, 128, 501, 519, 226, 487, 259, 568, 290, 239, 380, 160, 599, 412, 0,
            320, 664, 117, 634, 150, 210, 656, 647, 623, 402, 526, 628, 551, 540, 279, 61, 376, 589, 468, 86,
            2, 277, 132, 455, 483, 167, 76, 90, 610, 190, 437, 285, 542, 216, 245, 154, 326, 100, 525, 34,
            657, 544, 222, 262, 187, 444, 58, 39, 48, 204, 249, 119, 604, 159, 668, 415, 232, 646, 577, 452,
            177, 152, 5, 267, 399, 185, 21, 71, 427, 421, 406, 250, 57, 304, 356, 17, 40, 13, 225, 508, 432,
            401, 396, 607, 169, 311, 343, 82, 588, 27, 281, 113, 195, 400, 394, 414, 488, 494, 131, 619, 6,
            80, 651, 1, 408, 111, 622, 492, 382, 469, 493, 237, 325, 352, 85, 30, 517, 341, 498, 392, 65, 643,
            202, 22, 97, 342, 321, 143, 439, 38, 636, 601, 552, 137, 459, 603, 410, 335, 110, 270, 215, 417,
            496, 592, 491, 409, 670, 336, 151, 539, 575, 608, 460, 293, 447, 476, 107, 516, 609, 645, 633,
            662, 641, 367, 404, 523, 666, 573, 461, 527, 291, 59, 289, 161, 475, 456, 578, 166, 72, 133, 445,
            213, 118, 395, 106, 269, 124, 129, 260, 522, 32, 192, 510, 220, 438, 635, 563, 431, 224, 37, 503,
            567, 605, 247, 206, 489, 205, 419, 660, 583, 36, 569, 255, 502, 41, 153, 172, 56, 587, 658, 116,
            546, 434, 212, 449, 99, 580, 371, 373, 173, 465, 45, 372, 9, 582, 162, 390, 256, 139, 299, 454,
            196, 300, 79, 420, 156, 329, 618, 330, 18, 334, 292, 84, 365, 43, 648, 470, 297, 181, 339, 125,
            243, 284, 234, 63, 103, 331, 203, 571, 328, 446, 364, 188, 282, 451, 207, 422, 663, 200, 148, 351,
            482, 547, 531, 287, 473, 47, 83, 44, 265, 312, 238, 165, 462, 441, 295, 613, 433, 423, 199, 520,
            319, 586, 298, 7, 307, 560, 91, 286, 484, 253, 182, 254, 518, 397, 345, 620, 88, 381, 659, 317,
            168, 198, 278, 266, 26, 219, 268, 108, 176, 283, 368, 549, 532, 227, 60, 296, 562, 411, 20, 68,
            92, 158, 490, 405, 105, 310, 241],
        "roots": 671,
        "wave_size": 32,
        "w9_bits": 271_740,
        "w9_high": 476_290,
        "net_bits": 660_000,
        "unfinished_refute": 7}


def f1_checked(e: dict, spec: dict) -> bool:
    return not _integrity(e, spec) and sorted(r["index"] for r in e["roots"]) == sorted(spec["indices"])


def f1_bits(e: dict, spec: dict) -> int:
    opening = e["opening"]
    own = sum(r["bits"]["nodes"] for r in e["roots"])
    return opening["moves"] * 4 + opening["trivial"] + spec["library_bits"] + own


def f1_complete(e: dict, spec: dict) -> dict:
    if not f1_checked(e, spec):
        return {"verdict": "not_run"}
    missing = sum(1 for r in e["roots"] if not r["complete"])
    verdict = "supported" if missing == 0 else "refuted" if missing > spec["unfinished_refute"] else "inconclusive"
    return {"verdict": verdict, "unfinished": missing}


def f1_size(e: dict, spec: dict) -> dict:
    if not f1_checked(e, spec):
        return {"verdict": "not_run"}
    bits = f1_bits(e, spec)
    lower = not all(r["complete"] for r in e["roots"])
    verdict = ("refuted" if bits > spec["w9_high"] else "supported" if not lower and bits <= spec["w9_bits"]
               else "inconclusive")
    return {"verdict": verdict, "bits": bits, "lower_bound": lower}


def f1_net(e: dict, spec: dict) -> dict:
    if not f1_checked(e, spec):
        return {"verdict": "not_run"}
    bits = f1_bits(e, spec)
    lower = not all(r["complete"] for r in e["roots"])
    verdict = "refuted" if bits >= spec["net_bits"] else "supported" if not lower else "inconclusive"
    return {"verdict": verdict, "bits": bits, "lower_bound": lower}
