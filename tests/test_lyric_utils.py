
import pytest
from iantirta.audio import AudioSegment
from iantirta.models import ASRResult, TextTiming, WordTiming

from iantirta.karaoke.lyric_utils import timing_overlap, align2ref


@pytest.fixture
def transcribe_res():
    import numpy as np

    return ASRResult(
        texts=[
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(0.9800000000000001),
                        end=np.float64(1.42),
                        word="Kini",
                        score=0.975341796875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(1.56),
                        end=np.float64(2.4400000000000004),
                        word="serahkan",
                        score=0.94769287109375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(2.64),
                        end=np.float64(3.3000000000000003),
                        word="semua",
                        score=0.9482421875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(3.3400000000000003),
                        end=np.float64(3.5),
                        word="ujungnya",
                        score=0.002528369426727295,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(3.5),
                        end=np.float64(3.58),
                        word="pada",
                        score=0.003285646438598633,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(3.58),
                        end=np.float64(3.7),
                        word="takdir",
                        score=0.002308964729309082,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(3.7),
                        end=np.float64(3.7800000000000002),
                        word="Jika",
                        score=0.13639259338378906,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(3.7800000000000002),
                        end=np.float64(3.9600000000000004),
                        word="beruntung,",
                        score=5.629327562120226e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(3.9600000000000004),
                        end=np.float64(4.04),
                        word="maka",
                        score=3.069639205932617e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.04),
                        end=np.float64(4.14),
                        word="kisah",
                        score=1.8358230590820313e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.14),
                        end=np.float64(4.2),
                        word="ini",
                        score=1.0927518208821614e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.2),
                        end=np.float64(4.26),
                        word="tak",
                        score=2.086162567138672e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.26),
                        end=np.float64(4.34),
                        word="akan",
                        score=2.294778823852539e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.34),
                        end=np.float64(4.5),
                        word="berakhir",
                        score=1.080334186553955e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.5),
                        end=np.float64(4.6000000000000005),
                        word="Untuk",
                        score=2.2530555725097655e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.6000000000000005),
                        end=np.float64(4.74),
                        word="sejenak",
                        score=2.0946775163922993e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.760000000000001),
                        end=np.float64(4.9),
                        word="kumohon",
                        score=2.494880131312779e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.9),
                        end=np.float64(5.0600000000000005),
                        word="berhenti",
                        score=5.960464477539062e-07,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(5.0600000000000005),
                        end=np.float64(5.28),
                        word="perdebatkan",
                        score=1.7556277188387783e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(5.28),
                        end=np.float64(7.46),
                        word="yang",
                        score=1.2218952178955078e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(7.46),
                        end=np.float64(7.5200000000000005),
                        word="tak",
                        score=4.0729840596516924e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(7.5200000000000005),
                        end=np.float64(7.66),
                        word="penting",
                        score=1.4475413731166294e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(7.66),
                        end=np.float64(7.74),
                        word="Kita",
                        score=3.1441450119018555e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(7.74),
                        end=np.float64(7.9),
                        word="berdamai,",
                        score=1.7061829566955566e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(7.9),
                        end=np.float64(8.16),
                        word="berjabat",
                        score=5.811452865600586e-07,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(8.18),
                        end=np.float64(8.9),
                        word="tangan,",
                        score=1.2715657552083333e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(8.9),
                        end=np.float64(9.1),
                        word="tertawakan",
                        score=1.7344951629638673e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(9.1),
                        end=np.float64(9.48),
                        word="yang",
                        score=1.2218952178955078e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(9.48),
                        end=np.float64(9.540000000000001),
                        word="tak",
                        score=4.0729840596516924e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(9.540000000000001),
                        end=np.float64(9.620000000000001),
                        word="bisa",
                        score=1.2218952178955078e-06,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(9.620000000000001),
                        end=np.float64(9.780000000000001),
                        word="dibenahi",
                        score=5.438923835754395e-07,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(4.128),
                        end=np.float64(4.308),
                        word="Hanya",
                        score=0.03752574920654297,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(4.908),
                        end=np.float64(5.168),
                        word="pada",
                        score=0.12053298950195312,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(5.388),
                        end=np.float64(6.768000000000001),
                        word="takdir.",
                        score=0.5511881510416666,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(8.188),
                        end=np.float64(8.648),
                        word="Jika",
                        score=0.53692626953125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(8.788),
                        end=np.float64(9.228),
                        word="baru",
                        score=0.21684980392456055,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(9.368),
                        end=np.float64(9.748000000000001),
                        word="dong,",
                        score=0.08085393905639648,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(10.428),
                        end=np.float64(10.788),
                        word="maka",
                        score=0.51068115234375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(10.928),
                        end=np.float64(11.208),
                        word="kita",
                        score=0.37058258056640625,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(12.66),
                        end=np.float64(12.88),
                        word="Tak",
                        score=0.45592466990152997,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(13.040000000000001),
                        end=np.float64(13.38),
                        word="akan",
                        score=0.7884521484375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(13.48),
                        end=np.float64(14.200000000000001),
                        word="berakhir",
                        score=0.22727105021476746,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(14.26),
                        end=np.float64(14.42),
                        word="untuk",
                        score=0.12810821533203126,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(14.620000000000001),
                        end=np.float64(19.32),
                        word="selamanya.",
                        score=0.008875714408026801,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(20.896),
                        end=np.float64(21.196),
                        word="Mohon",
                        score=0.08500585556030274,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(21.296),
                        end=np.float64(21.976),
                        word="berhenti",
                        score=0.2476767897605896,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(22.116),
                        end=np.float64(22.556),
                        word="pada",
                        score=0.262451171875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(22.636),
                        end=np.float64(23.496000000000002),
                        word="bahkan",
                        score=0.2843360900878906,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(23.576),
                        end=np.float64(23.676000000000002),
                        word="yang",
                        score=0.03960752487182617,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(23.776),
                        end=np.float64(24.576),
                        word="tak",
                        score=0.20519469181696573,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(24.596),
                        end=np.float64(25.456000000000003),
                        word="penting",
                        score=0.20318331037248885,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(25.996000000000002),
                        end=np.float64(26.456000000000003),
                        word="kita",
                        score=0.87939453125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(26.556),
                        end=np.float64(27.576),
                        word="berdamai",
                        score=0.30313682556152344,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(28.216),
                        end=np.float64(28.976),
                        word="berjabat",
                        score=0.3941974639892578,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(29.036),
                        end=np.float64(29.716),
                        word="tangan.",
                        score=0.03811200459798177,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(33.364000000000004),
                        end=np.float64(33.984),
                        word="Bunuh",
                        score=0.16839923858642578,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(34.064),
                        end=np.float64(34.724000000000004),
                        word="hati.",
                        score=0.17470359802246094,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(35.5),
                        end=np.float64(35.86),
                        word="Anggap",
                        score=0.17552630106608072,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(35.94),
                        end=np.float64(36.34),
                        word="saja.",
                        score=0.17294979095458984,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(36.54),
                        end=np.float64(37.08),
                        word="Jabat",
                        score=0.614532470703125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(37.66),
                        end=np.float64(38.44),
                        word="tangan.",
                        score=0.1690343221028646,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(39.852),
                        end=np.float64(40.251999999999995),
                        word="cerita",
                        score=0.7394205729166666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(40.332),
                        end=np.float64(41.152),
                        word="seperti",
                        score=0.7124372209821429,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(41.372),
                        end=np.float64(41.652),
                        word="apa",
                        score=0.9947916666666666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(42.132),
                        end=np.float64(42.292),
                        word="yang",
                        score=0.9904296875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(42.391999999999996),
                        end=np.float64(42.852),
                        word="mau",
                        score=0.8870442708333334,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(43.192),
                        end=np.float64(43.412),
                        word="kau",
                        score=0.9716796875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(43.472),
                        end=np.float64(44.692),
                        word="kenang?",
                        score=0.8254656110491071,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(45.692),
                        end=np.float64(46.072),
                        word="Kita",
                        score=0.505615234375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(46.272),
                        end=np.float64(46.732),
                        word="buat",
                        score=0.39263916015625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(46.832),
                        end=np.float64(48.412),
                        word="malam",
                        score=0.03310738801956177,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(48.232),
                        end=np.float64(50.292),
                        word="Jadi",
                        score=0.4387531280517578,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(50.392),
                        end=np.float64(50.932),
                        word="saat",
                        score=0.25488853454589844,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(50.972),
                        end=np.float64(51.592),
                        word="waktunya",
                        score=0.1679391860961914,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(54.592),
                        end=np.float64(55.012),
                        word="Jika",
                        score=0.9080810546875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(55.332),
                        end=np.float64(55.592),
                        word="ada",
                        score=0.4306742350260417,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(55.692),
                        end=np.float64(55.892),
                        word="yang",
                        score=0.7923583984375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(55.952),
                        end=np.float64(56.592),
                        word="salah,",
                        score=0.5113151550292969,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(57.092),
                        end=np.float64(57.472),
                        word="jika",
                        score=0.93115234375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(57.612),
                        end=np.float64(57.832),
                        word="kau",
                        score=0.9708658854166666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(57.912),
                        end=np.float64(58.712),
                        word="kecewa,",
                        score=0.4130261739095052,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(59.012),
                        end=np.float64(59.532),
                        word="tolong",
                        score=0.1484521230061849,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(59.872),
                        end=np.float64(60.492000000000004),
                        word="jagalah.",
                        score=0.03478356770106724,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(63.472),
                        end=np.float64(63.912),
                        word="Kita",
                        score=0.445281982421875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(64.032),
                        end=np.float64(64.512),
                        word="buat",
                        score=0.20558929443359375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(64.612),
                        end=np.float64(65.05199999999999),
                        word="malam",
                        score=0.08359953165054321,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(67.92399999999999),
                        end=np.float64(68.384),
                        word="Bahkan",
                        score=0.32988739013671875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(68.484),
                        end=np.float64(68.824),
                        word="tak",
                        score=0.19183349609375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(68.86399999999999),
                        end=np.float64(69.044),
                        word="usah.",
                        score=0.07633209228515625,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(69.456),
                        end=np.float64(70.296),
                        word="Saya",
                        score=0.03722047805786133,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(70.656),
                        end=np.float64(71.57600000000001),
                        word="akan",
                        score=0.05049586296081543,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(72.376),
                        end=np.float64(73.07600000000001),
                        word="selagi",
                        score=0.36578623453776044,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(73.176),
                        end=np.float64(73.336),
                        word="kau",
                        score=0.5340908368428549,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(73.456),
                        end=np.float64(73.99600000000001),
                        word="masih",
                        score=0.3915273666381836,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(74.056),
                        end=np.float64(75.036),
                        word="bisa.",
                        score=0.47418212890625,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(76.864),
                        end=np.float64(77.26400000000001),
                        word="Kita",
                        score=0.6783103942871094,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(77.40400000000001),
                        end=np.float64(77.864),
                        word="buat",
                        score=0.757080078125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(77.98400000000001),
                        end=np.float64(79.84400000000001),
                        word="menenangkan.",
                        score=0.24140997366471725,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(81.7),
                        end=np.float64(81.92),
                        word="Ah.",
                        score=0.03716087341308594,
                        language=None,
                        
                    )
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(87.608),
                        end=np.float64(88.108),
                        word="Tiara",
                        score=0.5154510498046875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(88.22800000000001),
                        end=np.float64(88.468),
                        word="yang",
                        score=0.6420497894287109,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(88.488),
                        end=np.float64(89.06800000000001),
                        word="kau",
                        score=0.5263163248697916,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(89.268),
                        end=np.float64(90.328),
                        word="pasi,",
                        score=0.80078125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(92.06800000000001),
                        end=np.float64(92.608),
                        word="mungkin",
                        score=0.4260354723249163,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(92.64800000000001),
                        end=np.float64(93.14800000000001),
                        word="dalam",
                        score=0.639208984375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(93.188),
                        end=np.float64(93.408),
                        word="hit",
                        score=0.3687744140625,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(97.09599999999999),
                        end=np.float64(97.29599999999999),
                        word="Dan",
                        score=0.3472086588541667,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(97.416),
                        end=np.float64(97.81599999999999),
                        word="bisa",
                        score=0.7554931640625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(97.916),
                        end=np.float64(98.45599999999999),
                        word="jadi",
                        score=0.6424064636230469,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(99.31599999999999),
                        end=np.float64(99.73599999999999),
                        word="kita",
                        score=0.535858154296875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(99.836),
                        end=np.float64(100.396),
                        word="tak.",
                        score=0.40025583902994794,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(101.312),
                        end=np.float64(102.012),
                        word="Jangan",
                        score=0.11394755045572917,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(102.092),
                        end=np.float64(102.512),
                        word="takut,",
                        score=0.06073455810546875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(103.012),
                        end=np.float64(103.852),
                        word="jangan",
                        score=0.2261505126953125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(104.332),
                        end=np.float64(105.312),
                        word="takut,",
                        score=0.09324188232421875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(105.372),
                        end=np.float64(105.692),
                        word="jangan",
                        score=0.16538783482142858,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(105.712),
                        end=np.float64(107.27199999999999),
                        word="takut.",
                        score=0.08260040283203125,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(108.952),
                        end=np.float64(109.472),
                        word="Terima",
                        score=0.5468368530273438,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(109.63199999999999),
                        end=np.float64(111.43199999999999),
                        word="kasih",
                        score=0.7536270141601562,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(112.612),
                        end=np.float64(112.85199999999999),
                        word="karena",
                        score=0.661425252755483,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(112.93199999999999),
                        end=np.float64(113.93199999999999),
                        word="mengenalmu.",
                        score=0.5352128028869629,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(117.22399999999999),
                        end=np.float64(117.50399999999999),
                        word="Aku",
                        score=0.9599609375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(117.664),
                        end=np.float64(117.804),
                        word="yang",
                        score=0.9967041015625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(117.86399999999999),
                        end=np.float64(118.344),
                        word="seperti",
                        score=0.8399832589285714,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(118.604),
                        end=np.float64(119.044),
                        word="ini.",
                        score=0.9353841145833334,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(122.2),
                        end=np.float64(122.56),
                        word="Anggap",
                        score=0.8203822544642857,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(122.66000000000001),
                        end=np.float64(123.38000000000001),
                        word="saja",
                        score=0.72088623046875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(123.5),
                        end=np.float64(124.32000000000001),
                        word="besok",
                        score=0.6297119140625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(124.44),
                        end=np.float64(124.74000000000001),
                        word="ini",
                        score=0.6968587239583334,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(124.86),
                        end=np.float64(125.9),
                        word="semuanya,",
                        score=0.4296379089355469,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(126.52000000000001),
                        end=np.float64(127.0),
                        word="semoga",
                        score=0.3221842447916667,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(127.12),
                        end=np.float64(127.62),
                        word="yang",
                        score=0.58779296875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(127.66),
                        end=np.float64(127.9),
                        word="kau",
                        score=0.8990885416666666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(128.12),
                        end=np.float64(128.48000000000002),
                        word="ingat",
                        score=0.7397054036458334,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(128.8),
                        end=np.float64(129.34),
                        word="hanya.",
                        score=0.3443450927734375,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(132.392),
                        end=np.float64(132.852),
                        word="Kita",
                        score=0.65631103515625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(132.932),
                        end=np.float64(133.37199999999999),
                        word="buat",
                        score=0.4221611022949219,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(136.82399999999998),
                        end=np.float64(137.084),
                        word="This",
                        score=0.18250274658203125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(137.20399999999998),
                        end=np.float64(137.384),
                        word="is",
                        score=0.329345703125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(137.384),
                        end=np.float64(137.664),
                        word="how",
                        score=0.044820149739583336,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(137.744),
                        end=np.float64(137.76399999999998),
                        word="I",
                        score=0.00560760498046875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(137.904),
                        end=np.float64(138.064),
                        word="do",
                        score=0.034885406494140625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(138.144),
                        end=np.float64(138.184),
                        word="it.",
                        score=0.028651952743530273,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(141.256),
                        end=np.float64(141.736),
                        word="Jika",
                        score=0.866943359375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(142.016),
                        end=np.float64(142.256),
                        word="ada",
                        score=0.232818603515625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(142.376),
                        end=np.float64(142.55599999999998),
                        word="yang",
                        score=0.507476806640625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(142.61599999999999),
                        end=np.float64(143.256),
                        word="salah,",
                        score=0.579559326171875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(143.756),
                        end=np.float64(144.136),
                        word="jika",
                        score=0.771728515625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(144.27599999999998),
                        end=np.float64(144.49599999999998),
                        word="kau",
                        score=0.5056559244791666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(144.596),
                        end=np.float64(144.916),
                        word="kecewa.",
                        score=0.08985519409179688,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(146.556),
                        end=np.float64(147.456),
                        word="Makan,",
                        score=0.41121902465820315,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(150.136),
                        end=np.float64(150.576),
                        word="kita",
                        score=0.34626007080078125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(150.696),
                        end=np.float64(151.176),
                        word="buat",
                        score=0.443603515625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(151.276),
                        end=np.float64(151.596),
                        word="makan.",
                        score=0.14641129970550537,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(152.688),
                        end=np.float64(154.628),
                        word="Tak",
                        score=0.10166422526041667,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(154.708),
                        end=np.float64(155.048),
                        word="akan",
                        score=0.8116455078125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(155.16799999999998),
                        end=np.float64(155.528),
                        word="tahu",
                        score=0.9766845703125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(155.66799999999998),
                        end=np.float64(156.16799999999998),
                        word="sampai",
                        score=0.920654296875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(156.248),
                        end=np.float64(157.688),
                        word="kapan.",
                        score=0.6766357421875,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(160.848),
                        end=np.float64(161.68800000000002),
                        word="Bisa",
                        score=0.060181617736816406,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(163.488),
                        end=np.float64(163.92800000000003),
                        word="kita",
                        score=0.3063201904296875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(164.068),
                        end=np.float64(164.52800000000002),
                        word="buat",
                        score=0.2971992492675781,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(164.568),
                        end=np.float64(164.92800000000003),
                        word="dunia.",
                        score=0.1081024169921875,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(166.244),
                        end=np.float64(166.404),
                        word="Aku",
                        score=0.0023943583170572915,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(166.60399999999998),
                        end=np.float64(167.304),
                        word="takkan",
                        score=0.023565292358398438,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(167.464),
                        end=np.float64(169.004),
                        word="pernah",
                        score=0.007426102956136067,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(169.10399999999998),
                        end=np.float64(173.684),
                        word="menyerah.",
                        score=0.023884296417236328,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(174.96),
                        end=np.float64(175.78),
                        word="Oke.",
                        score=0.17228190104166666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(176.62),
                        end=np.float64(177.74),
                        word="Oke.Oke",
                        score=0.174041748046875,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(180.132),
                        end=np.float64(181.33200000000002),
                        word="Jadi.",
                        score=0.186492919921875,
                        language=None,
                        
                    )
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(183.428),
                        end=np.float64(183.56799999999998),
                        word="Aku",
                        score=0.0064512888590494795,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(184.208),
                        end=np.float64(184.628),
                        word="takkan",
                        score=0.033232688903808594,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(184.688),
                        end=np.float64(185.128),
                        word="pernah",
                        score=0.015306393305460611,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(185.468),
                        end=np.float64(186.22799999999998),
                        word="menyerah.",
                        score=0.008959531784057617,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(188.95999999999998),
                        end=np.float64(189.22),
                        word="Aku",
                        score=0.027959187825520832,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(189.35999999999999),
                        end=np.float64(190.48),
                        word="takkan",
                        score=0.02827898661295573,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(190.56),
                        end=np.float64(191.28),
                        word="pernah",
                        score=0.01624925931294759,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(191.35999999999999),
                        end=np.float64(192.04),
                        word="menyerah.",
                        score=0.018141984939575195,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(194.568),
                        end=np.float64(194.988),
                        word="Kita",
                        score=0.9974365234375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(195.168),
                        end=np.float64(195.648),
                        word="buat",
                        score=0.910400390625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(195.748),
                        end=np.float64(196.108),
                        word="menyanyi.",
                        score=0.3744751214981079,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(196.92000000000002),
                        end=np.float64(197.3),
                        word="Angka",
                        score=0.5213750839233399,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(199.04),
                        end=np.float64(199.44),
                        word="desa",
                        score=0.48602867126464844,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(199.58),
                        end=np.float64(200.3),
                        word="waktu",
                        score=0.5977176547050476,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(200.48),
                        end=np.float64(200.68),
                        word="yang",
                        score=0.982666015625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(200.78),
                        end=np.float64(202.28),
                        word="marah.",
                        score=0.38828215599060056,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(203.856),
                        end=np.float64(204.036),
                        word="Jika",
                        score=0.20009422302246094,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(204.296),
                        end=np.float64(204.516),
                        word="ada",
                        score=0.3407643636067708,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(204.636),
                        end=np.float64(204.796),
                        word="yang",
                        score=0.995361328125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(204.896),
                        end=np.float64(205.456),
                        word="salah,",
                        score=0.93017578125,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(205.976),
                        end=np.float64(206.356),
                        word="jika",
                        score=0.9990234375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(206.516),
                        end=np.float64(206.716),
                        word="kau",
                        score=0.9938151041666666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(206.816),
                        end=np.float64(207.536),
                        word="kecewa,",
                        score=0.586181640625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(207.916),
                        end=np.float64(208.436),
                        word="tolong",
                        score=0.5391031901041666,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(208.77599999999998),
                        end=np.float64(213.736),
                        word="maaf.",
                        score=0.026260018348693848,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(210.392),
                        end=np.float64(212.792),
                        word="Kita",
                        score=0.6761474609375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(212.912),
                        end=np.float64(213.412),
                        word="buat",
                        score=0.535369873046875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(213.472),
                        end=np.float64(213.752),
                        word="manusia.",
                        score=0.1681812150137765,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(216.80800000000002),
                        end=np.float64(217.268),
                        word="Takkan",
                        score=0.7996075948079427,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(217.368),
                        end=np.float64(217.788),
                        word="tahu",
                        score=0.7371231317520142,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(217.928),
                        end=np.float64(218.388),
                        word="sampai",
                        score=0.982421875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(218.46800000000002),
                        end=np.float64(220.508),
                        word="kapan,",
                        score=0.6219482421875,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(221.268),
                        end=np.float64(221.948),
                        word="selagi",
                        score=0.11066945393880208,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(222.048),
                        end=np.float64(222.228),
                        word="kau",
                        score=0.10478798548380534,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(222.268),
                        end=np.float64(222.62800000000001),
                        word="masih",
                        score=0.01268939971923828,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(222.712),
                        end=np.float64(222.852),
                        word="Hari",
                        score=0.08849740028381348,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(222.972),
                        end=np.float64(223.952),
                        word="biasa.",
                        score=0.12914810180664063,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(226.344),
                        end=np.float64(226.824),
                        word="Buat",
                        score=0.1066741943359375,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(227.084),
                        end=np.float64(228.864),
                        word="menantang",
                        score=0.24857913123236763,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(230.144),
                        end=np.float64(230.844),
                        word="selagi",
                        score=0.4801247914632161,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(230.944),
                        end=np.float64(231.104),
                        word="kau",
                        score=0.624857763449351,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(231.264),
                        end=np.float64(231.784),
                        word="masih",
                        score=0.7084222793579101,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(231.824),
                        end=np.float64(232.784),
                        word="bisa.",
                        score=0.6923828125,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(234.616),
                        end=np.float64(235.01600000000002),
                        word="Kita",
                        score=0.5304336547851562,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(235.19600000000003),
                        end=np.float64(235.596),
                        word="buat",
                        score=0.9664306640625,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(235.976),
                        end=np.float64(237.276),
                        word="menangga.",
                        score=0.4034919738769531,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
            TextTiming(
                words=[
                    WordTiming(
                        start=np.float64(240.41600000000003),
                        end=np.float64(241.29600000000002),
                        word="Aku",
                        score=0.018666585286458332,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(241.39600000000002),
                        end=np.float64(244.836),
                        word="takkan",
                        score=0.05988883972167969,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(246.156),
                        end=np.float64(246.89600000000002),
                        word="pernah",
                        score=0.02181212107340495,
                        language=None,
                        
                    ),
                    WordTiming(
                        start=np.float64(247.316),
                        end=np.float64(248.616),
                        word="menyerah.",
                        score=0.017052173614501953,
                        language=None,
                        
                    ),
                ],
                language="id",
                ass_event=None,
                
                _text=None,
            ),
        ]
    )


@pytest.fixture
def indo_lyric() -> str:
    return """Kini serahkan semua ujungnya pada takdir
Jika beruntung maka kisah ini tak akan berakhir
Untuk sejenak kumohon berhenti perdebatkan yang tak penting
Kita berdamai berjabat tangan
Tertawakan yang tak bisa dibenahi
Anggap saja besok ini semua hilang
C'rita s'perti apa yang mau kau kenang
Kita buat menyenangkan
Di sisa waktu yang ada
Jika ada yang salah jika kau kecewa
Tolong maafkan
Kita buat menyenangkan
Takkan tahu sampai kapan
Selagiku masih bisa
Kita buat menyenangkan
Tiada yang tahu pasti
Mungkin dalam hitungan hari semua berhenti
Dan bisa jadi kita tak ada lagi
S'belum terlambat kuucapkan banyak terima kasih
Karena mengenalmu buatku jadi
Aku yang s'perti ini
Anggap saja besok ini semua hilang
S'moga yang kau ingat hanya yang baiknya
Kita buat menyenangkan
Di sisa waktu yang ada
Jika ada yang salah jika kau kecewa
Tolong maafkan
Kita buat menyenangkan
Takkan tahu sampai kapan
Selagiku masih bisa
Kita buat menyenangkan
Kita buat menyenangkan
Di sisa waktu yang ada
Jika ada yang salah jika kau kecewa
Tolong maafkan
Kita buat menyenangkan
Takkan tahu sampai kapan
Selagiku masih bisa
Kita buat menyenangkan
Selagiku masih bisa
Kita buat menyenangkan
"""


@pytest.fixture
def audio_segments() -> list[AudioSegment]:
    import numpy as np

    return [
        AudioSegment(start=np.float64(0.16), end=np.float64(3.808)),
        AudioSegment(start=np.float64(4.128), end=np.float64(7.168)),
        AudioSegment(start=np.float64(7.488), end=np.float64(11.248)),
        AudioSegment(start=np.float64(12.56), end=np.float64(19.344)),
        AudioSegment(start=np.float64(20.896), end=np.float64(30.784)),
        AudioSegment(start=np.float64(33.264), end=np.float64(34.944)),
        AudioSegment(start=np.float64(34.96), end=np.float64(36.384)),
        AudioSegment(start=np.float64(36.4), end=np.float64(38.576)),
        AudioSegment(start=np.float64(38.592), end=np.float64(47.136)),
        AudioSegment(start=np.float64(47.792), end=np.float64(51.632)),
        AudioSegment(start=np.float64(51.872), end=np.float64(60.528)),
        AudioSegment(start=np.float64(61.632), end=np.float64(64.912)),
        AudioSegment(start=np.float64(65.904), end=np.float64(69.088)),
        AudioSegment(start=np.float64(69.456), end=np.float64(76.128)),
        AudioSegment(start=np.float64(76.144), end=np.float64(79.872)),
        AudioSegment(start=np.float64(80.8), end=np.float64(84.64)),
        AudioSegment(start=np.float64(85.248), end=np.float64(93.52)),
        AudioSegment(start=np.float64(95.856), end=np.float64(100.496)),
        AudioSegment(start=np.float64(101.312), end=np.float64(106.048)),
        AudioSegment(start=np.float64(108.832), end=np.float64(114.256)),
        AudioSegment(start=np.float64(116.624), end=np.float64(119.904)),
        AudioSegment(start=np.float64(120.48), end=np.float64(129.376)),
        AudioSegment(start=np.float64(130.272), end=np.float64(133.52)),
        AudioSegment(start=np.float64(134.784), end=np.float64(138.224)),
        AudioSegment(start=np.float64(138.576), end=np.float64(144.944)),
        AudioSegment(start=np.float64(146.256), end=np.float64(151.584)),
        AudioSegment(start=np.float64(152.528), end=np.float64(159.36)),
        AudioSegment(start=np.float64(160.848), end=np.float64(164.96)),
        AudioSegment(start=np.float64(166.224), end=np.float64(172.4)),
        AudioSegment(start=np.float64(173.68), end=np.float64(179.088)),
        AudioSegment(start=np.float64(179.472), end=np.float64(182.672)),
        AudioSegment(start=np.float64(183.408), end=np.float64(187.376)),
        AudioSegment(start=np.float64(188.32), end=np.float64(192.416)),
        AudioSegment(start=np.float64(192.768), end=np.float64(196.144)),
        AudioSegment(start=np.float64(196.56), end=np.float64(203.648)),
        AudioSegment(start=np.float64(203.856), end=np.float64(209.216)),
        AudioSegment(start=np.float64(209.392), end=np.float64(213.792)),
        AudioSegment(start=np.float64(214.448), end=np.float64(222.656)),
        AudioSegment(start=np.float64(222.672), end=np.float64(224.944)),
        AudioSegment(start=np.float64(226.304), end=np.float64(233.84)),
        AudioSegment(start=np.float64(234.336), end=np.float64(239.36)),
        AudioSegment(start=np.float64(240.336), end=np.float64(250.016)),
    ]


@pytest.mark.parametrize(
    ('aseg', 'word', 'expected'),
    [
        # Fit in between
        ((0.0, 1.0), (0.02, 0.98), True),
        # Fully Over
        ((1.0, 2.0), (0.98, 2.1), True),
        # Left overlap
        ((1.0, 2.0), (0.5, 1.4), True),
        # Right overlap
        ((2.0, 2.5), (2.2, 2.9), True),
        # Equal
        ((2.0, 3.0), (2.0, 3.0), True),
        # Out
        ((2.0, 2.99), (3.0, 4.0), False),
    ]
)
def test_timing_overlap(aseg, word, expected):
    aseg = AudioSegment(
        start=aseg[0],
        end=aseg[1]
    )
    word = WordTiming(
        start=word[0],
        end=word[1],
        word="",
        score=None
    )
    assert timing_overlap(aseg, word) == expected


def test_compare_preparation_for_sequence_align(transcribe_res, indo_lyric, audio_segments):
    # Cust
    import iantirta.text as text_utils
    lyric_words = [
        (word, norm)
        for line in indo_lyric.splitlines()
        if (_line := line.strip())
        and not _line.startswith("[")
        for word in _line.split()
        if (norm := text_utils.normalize(word, kaldi=True).strip())
    ]
    clean_lyric_words = [norm for _, norm in lyric_words]

    valid_hyp_words = [
        (w, norm, text.language)
        for text in transcribe_res.texts
        for w in text.words
        if w.score > 0.1
        and (norm := text_utils.normalize(w.clean, kaldi=True).strip())
    ]
    clean_hypothesis = [norm for _, norm, _ in valid_hyp_words]

    # Kplus
    from kplus.pipelines.lyrics.utils import Tokens
    from kplus.tools.text import similarity
    ref_tokens = Tokens.from_reference(indo_lyric)
    hyp_tokens = Tokens.from_asr(transcribe_res.texts)


    # Validate
    from sequence_align.pairwise import needleman_wunsch_with_scores as nwws
    def mwws_score(a:str, b: str) -> float:
        if a == b:
            return 2.0 # Match exactly
        if similarity(a, b) > 0.6:
            return 1.0
        return -3.0
    
    def mwws_score2(a:str, b: str) -> float:
        if a == b:
            return 2.0 # Match exactly
        if text_utils.similarity(a, b) > 0.6:
            return 1.0
        return -3.0

    k_refs, k_hyps = nwws(
        ref_tokens.cleans, hyp_tokens.cleans,
        gap="-", score_fn=mwws_score, # Winner
        indel_score=-1
    )
    refs, hyps = nwws(
        clean_lyric_words, clean_hypothesis,
        gap="-", score_fn=mwws_score2,
        indel_score=-1
    )
    for i, (r1, r2) in enumerate(zip(k_refs, refs)):
        _r1 = k_refs[i-3:i+3]
        _r2 = refs[i-3:i+3]
        assert r1 == r2, f"{_r1} -> {_r2} | {k_hyps[i-3:i+1]} -> {hyps[i-3:i+1]}"

    assert hyps == k_hyps


def test_compare_sequence_align(transcribe_res, indo_lyric):
    # Cust
    import iantirta.text as text_utils
    lyric_words = [
        (word, norm)
        for line in indo_lyric.splitlines()
        if (_line := line.strip())
        and not _line.startswith("[")
        for word in _line.split()
        if (norm := text_utils.normalize(word, kaldi=True).strip())
    ]
    clean_lyric_words = [norm for _, norm in lyric_words]

    valid_hyp_words = [
        (w, norm, text.language)
        for text in transcribe_res.texts
        for w in text.words
        if w.score > 0.1
        and (norm := text_utils.normalize(w.clean, kaldi=True).strip())
    ]
    clean_hypothesis = [norm for _, norm, _ in valid_hyp_words]

    # Kplus
    from kplus.pipelines.lyrics.utils import Tokens
    from kplus.tools.text import similarity
    ref_tokens = Tokens.from_reference(indo_lyric)
    hyp_tokens = Tokens.from_asr(transcribe_res.texts)

    from kplus.pipelines.lyrics.sequence_aligner import SequenceAligner
    k_alignments = SequenceAligner(False).sequence_align(ref_tokens, hyp_tokens)

    alignments = text_utils.align(
        clean_lyric_words,
        clean_hypothesis
    )

    for op, a in zip(k_alignments.operations, alignments):
        _op = a.operation
        if _op == "equal": _op = "match"
        assert op["op"] == _op, (
            f"{a.left} - {a.right}\n"
            f'{ref_tokens[op["ref_idx"]]} = {hyp_tokens[op["hyp_idx"]] if op["hyp_idx"] else "-"}'
        )


def test_compare_align_lyric(transcribe_res, indo_lyric, audio_segments):
    from pprint import pprint

    from kplus.pipelines.lyrics import align2ref as kplus_align2ref

    ref_result, new_audio_segments = align2ref(
        hypothesis=transcribe_res,
        lyric=indo_lyric,
        audio_segments=audio_segments,
    )
    kplus_ref_result, kplus_new_audio_segments = kplus_align2ref(
        hypothesis=transcribe_res,
        reference=indo_lyric,
        audiosegments=audio_segments,
        raise_if_not_reliable=False
    )

    for text, ktext in zip(ref_result.texts, kplus_ref_result.texts):
        print(text.end, ktext.end)
        assert text.text == ktext.text


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([
        '-v', '-s',
        __file__ + "::test_compare_sequence_align"
    ]))
