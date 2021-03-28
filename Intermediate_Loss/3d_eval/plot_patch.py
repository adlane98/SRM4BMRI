import matplotlib.pyplot as plt
import numpy as np

inp = [3.690720319747925, 5.536080479621887, 5.536080479621887, 3.690720319747925, 7.38144063949585, 12.917521119117737, 9.226800799369812, 11.072160959243774
, 0.0, 1.8453601598739624, 7.38144063949585, 7.38144063949585, 14.7628812789917, 12.917521119117737, 5.536080479621887, 9.226800799369812, 3.690720319747925, 
9.226800799369812, 7.38144063949585, 9.226800799369812, 14.7628812789917, 3.690720319747925, 9.226800799369812, 5.536080479621887, 7.38144063949585, 3.690720319747925, 
9.226800799369812, 9.226800799369812, 5.536080479621887, 9.226800799369812, 7.38144063949585, 3.690720319747925, 9.226800799369812, 9.226800799369812, 86.73192751407623, 
7.38144063949585, 9.226800799369812, 11.072160959243774, 12.917521119117737, 57.206164956092834, 167.92777454853058, 374.60811245441437, 16.60824143886566, 79.35048687458038, 
188.22673630714417, 313.7112271785736, 398.5977945327759, 518.5462049245834, 489.02044236660004]

gt = [0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0,
 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 14.7628812789917, 14.7628812789917, 7.38144063949585, 14.7628812789917,
 0.0, 22.14432191848755, 7.38144063949585, 14.7628812789917, 7.38144063949585, 51.67008447647095, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 22.14432191848755, 
 7.38144063949585, 22.14432191848755, 22.14432191848755, 7.38144063949585, 14.7628812789917, 7.38144063949585, 22.14432191848755, 29.5257625579834, 103.3401689529419, 7.38144063949585,
  7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 14.7628812789917, 7.38144063949585, 22.14432191848755,
   36.90720319747925, 103.3401689529419, 199.29889726638794, 7.38144063949585, 0.0, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585,
    22.14432191848755, 7.38144063949585, 51.67008447647095, 147.628812789917, 228.82465982437134, 383.8349132537842, 22.14432191848755, 7.38144063949585, 7.38144063949585, 7.38144063949585, 
    14.7628812789917, 7.38144063949585, 29.5257625579834, 7.38144063949585, 14.7628812789917, 51.67008447647095, 169.77313470840454, 199.29889726638794, 354.3091506958008, 568.3709292411804,
     7.38144063949585, 0.0, 0.0, 14.7628812789917, 22.14432191848755, 14.7628812789917, 0.0, 22.14432191848755, 66.43296575546265, 191.9174566268921, 228.82465982437134, 354.3091506958008,
      546.2266073226929, 590.515251159668, 14.7628812789917, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 14.7628812789917, 81.19584703445435, 199.29889726638794, 
      243.58754110336304, 339.5462694168091, 457.6493196487427, 509.3194041252136, 376.45347261428833, 0.0, 14.7628812789917, 14.7628812789917, 7.38144063949585, 7.38144063949585,
       14.7628812789917, 73.8144063949585, 191.9174566268921, 302.63906621932983, 383.8349132537842, 450.2678790092468, 531.4637260437012, 479.7936415672302, 236.2061004638672, 
       14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 14.7628812789917, 81.19584703445435, 177.1545753479004, 295.257625579834, 398.5977945327759, 442.886438369751, 
       531.4637260437012, 553.6080479621887, 310.0205068588257, 250.9689817428589, 22.14432191848755, 0.0, 14.7628812789917, 22.14432191848755, 110.72160959243774, 221.4432191848755, 
       354.3091506958008, 435.5049977302551, 457.6493196487427, 553.6080479621887, 597.8966917991638, 346.92771005630493, 236.2061004638672, 376.45347261428833, 7.38144063949585,
        73.8144063949585, 140.24737215042114, 228.82465982437134, 310.0205068588257, 405.97923517227173, 516.7008447647095, 516.7008447647095, 568.3709292411804, 590.515251159668, 
        391.21635389328003, 236.2061004638672, 376.45347261428833, 398.5977945327759, 243.58754110336304, 317.40194749832153, 376.45347261428833, 428.1235570907593, 457.6493196487427, 
        487.1750822067261, 501.9379634857178, 531.4637260437012, 524.0822854042053, 310.0205068588257, 236.2061004638672, 376.45347261428833, 405.97923517227173, 369.0720319747925, 
        428.1235570907593, 450.2678790092468, 450.2678790092468, 568.3709292411804, 620.0410137176514, 494.5565228462219, 472.4122009277344, 531.4637260437012, 391.21635389328003,
         250.9689817428589, 376.45347261428833, 428.1235570907593, 383.8349132537842, 346.92771005630493]

gt = [369.0720319747925, 376.45347261428833, 391.21635389328003, 413.3606758117676, 420.7421164512634, 398.5977945327759, 317.40194749832153, 221.4432191848755, 73.8144063949585, 140.24737215042114, 354.3091506958008, 501.9379634857178, 435.5049977302551, 177.1545753479004, 361.69059133529663, 354.3091506958008, 361.69059133529663, 369.0720319747925, 383.8349132537842, 405.97923517227173, 413.3606758117676, 405.97923517227173, 369.0720319747925, 295.257625579834, 125.48449087142944, 147.628812789917, 391.21635389328003, 494.5565228462219, 405.97923517227173, 391.21635389328003, 376.45347261428833, 369.0720319747925, 361.69059133529663, 376.45347261428833, 383.8349132537842, 398.5977945327759, 420.7421164512634, 420.7421164512634, 383.8349132537842, 324.7833881378174, 162.3916940689087, 125.48449087142944, 383.8349132537842, 391.21635389328003, 376.45347261428833, 354.3091506958008, 376.45347261428833, 405.97923517227173, 398.5977945327759, 391.21635389328003, 398.5977945327759, 413.3606758117676, 413.3606758117676, 391.21635389328003, 354.3091506958008, 250.9689817428589, 369.0720319747925, 369.0720319747925, 361.69059133529663, 354.3091506958008, 383.8349132537842, 398.5977945327759, 391.21635389328003, 398.5977945327759, 391.21635389328003, 369.0720319747925, 369.0720319747925, 391.21635389328003, 405.97923517227173, 413.3606758117676, 346.92771005630493, 361.69059133529663, 369.0720319747925, 376.45347261428833, 369.0720319747925, 369.0720319747925, 376.45347261428833, 383.8349132537842, 383.8349132537842, 376.45347261428833, 383.8349132537842, 369.0720319747925, 354.3091506958008, 376.45347261428833, 346.92771005630493, 376.45347261428833, 391.21635389328003, 398.5977945327759, 383.8349132537842, 354.3091506958008, 361.69059133529663, 391.21635389328003, 391.21635389328003, 376.45347261428833, 361.69059133529663, 369.0720319747925, 383.8349132537842, 391.21635389328003, 346.92771005630493, 398.5977945327759, 391.21635389328003, 391.21635389328003, 398.5977945327759, 376.45347261428833, 361.69059133529663, 383.8349132537842, 383.8349132537842, 369.0720319747925, 369.0720319747925, 369.0720319747925, 383.8349132537842, 383.8349132537842, 221.4432191848755, 295.257625579834, 369.0720319747925, 369.0720319747925, 354.3091506958008, 361.69059133529663, 391.21635389328003, 391.21635389328003, 369.0720319747925, 369.0720319747925, 369.0720319747925, 361.69059133529663, 369.0720319747925, 369.0720319747925, 428.1235570907593, 287.87618494033813, 280.4947443008423, 346.92771005630493, 376.45347261428833, 398.5977945327759, 413.3606758117676, 391.21635389328003, 376.45347261428833, 391.21635389328003, 376.45347261428833, 361.69059133529663, 361.69059133529663, 346.92771005630493, 295.257625579834, 243.58754110336304, 191.9174566268921, 228.82465982437134, 310.0205068588257, 369.0720319747925, 391.21635389328003, 420.7421164512634, 450.2678790092468, 435.5049977302551, 346.92771005630493, 310.0205068588257, 354.3091506958008, 369.0720319747925, 147.628812789917, 88.5772876739502, 51.67008447647095, 59.0515251159668, 81.19584703445435, 118.1030502319336, 162.3916940689087, 199.29889726638794, 265.7318630218506, 332.16482877731323, 310.0205068588257, 324.7833881378174, 405.97923517227173, 376.45347261428833, 295.257625579834, 258.35042238235474, 243.58754110336304, 236.2061004638672, 228.82465982437134, 214.06177854537964, 206.6803379058838, 140.24737215042114, 81.19584703445435, 59.0515251159668, 132.8659315109253, 332.16482877731323, 509.3194041252136, 457.6493196487427, 280.4947443008423, 398.5977945327759, 420.7421164512634, 361.69059133529663, 361.69059133529663, 391.21635389328003, 391.21635389328003, 383.8349132537842, 339.5462694168091, 206.6803379058838, 265.7318630218506, 435.5049977302551, 516.7008447647095, 465.0307602882385]
inp = [369.9938809619057, 380.19775102984545, 391.4602695004395, 349.89374695816946, 272.4876144958, 286.1973263119737, 346.20076194102205, 382.83782117177225, 370.8462349861878, 382.127961750342, 394.1037453175783, 389.9580043514227, 347.13882160417893, 288.6164354696919, 366.8227219799529, 370.6647354252827, 378.98329642609883, 386.5566553270503, 384.4128474802893, 378.7074027120672, 374.9697305136516, 359.12282871939533, 383.1298390134861, 375.94486973335927, 377.31396946614836, 378.3739924106988, 370.5134968409425, 378.73137119591456, 309.3104051863448, 329.0285791437145, 367.0685995343465, 388.3727724640596, 382.7030560756948, 365.76342212777706, 364.1242749412977, 214.22037252446168, 185.9016497293059, 242.1548815893785, 293.8663641095786, 325.87444671962953, 331.636339157731, 379.88086169754735, 279.0266861524648, 269.9010708927072, 262.70330456410875, 249.0123229912226, 193.89775244329633, 298.34564569239944, 457.80169855483757]

gt = [7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 0.0, 14.7628812789917, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 0.0, 22.14432191848755, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 0.0, 0.0, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 22.14432191848755, 0.0, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 14.7628812789917, 14.7628812789917, 14.7628812789917, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 22.14432191848755, 0.0, 7.38144063949585, 0.0, 7.38144063949585, 14.7628812789917, 22.14432191848755, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 22.14432191848755, 0.0, 7.38144063949585, 14.7628812789917, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 22.14432191848755, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 22.14432191848755, 14.7628812789917, 14.7628812789917, 0.0, 7.38144063949585, 7.38144063949585, 14.7628812789917, 0.0, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 22.14432191848755, 29.5257625579834, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 14.7628812789917, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 0.0, 7.38144063949585, 14.7628812789917, 0.0, 7.38144063949585, 14.7628812789917, 14.7628812789917, 14.7628812789917, 14.7628812789917, 7.38144063949585]
inp = [7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 0.0, 11.072160959243774, 0.0, 14.7628812789917, 7.38144063949585, 7.38144063949585, 3.690720319747925, 7.38144063949585, 11.072160959243774, 11.072160959243774, 3.690720319747925, 3.690720319747925, 3.690720319747925, 11.072160959243774, 7.38144063949585, 3.690720319747925, 3.690720319747925, 3.690720319747925, 11.072160959243774, 7.38144063949585, 11.072160959243774, 11.072160959243774, 3.690720319747925, 0.0, 3.690720319747925, 7.38144063949585, 3.690720319747925, 3.690720319747925, 3.690720319747925, 11.072160959243774, 7.38144063949585, 7.38144063949585, 11.072160959243774, 3.690720319747925, 11.072160959243774, 0.0, 11.072160959243774, 0.0, 14.7628812789917, 3.690720319747925, 3.690720319747925, 3.690720319747925, 3.690720319747925, 7.38144063949585, 14.7628812789917, 14.7628812789917, 11.072160959243774, 7.38144063949585, 3.690720319747925, 11.072160959243774, 3.690720319747925, 11.072160959243774, 14.7628812789917, 11.072160959243774, 11.072160959243774, 3.690720319747925, 11.072160959243774, 11.072160959243774, 3.690720319747925, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 14.7628812789917, 14.7628812789917, 3.690720319747925, 11.072160959243774, 3.690720319747925, 11.072160959243774, 3.690720319747925, 25.835042238235474, 11.072160959243774, 3.690720319747925, 7.38144063949585, 11.072160959243774, 3.690720319747925, 3.690720319747925, 11.072160959243774, 11.072160959243774, 3.690720319747925, 11.072160959243774, 7.38144063949585, 11.072160959243774, 3.690720319747925, 3.690720319747925, 7.38144063949585, 11.072160959243774, 14.7628812789917, 11.072160959243774]

inp = [3.690720319747925, 7.38144063949585, 3.690720319747925, 7.38144063949585, 3.690720319747925, 0.0, 7.38144063949585, 3.690720319747925, 7.38144063949585, 0.0, 7.38144063949585, 0.0, 3.690720319747925, 14.7628812789917, 3.690720319747925, 3.690720319747925, 0.0, 14.7628812789917, 7.38144063949585, 0.0, 14.7628812789917, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 3.690720319747925, 11.072160959243774, 3.690720319747925, 7.38144063949585, 7.38144063949585, 7.38144063949585, 11.072160959243774, 7.38144063949585, 7.38144063949585, 7.38144063949585, 11.072160959243774, 0.0, 7.38144063949585, 3.690720319747925, 7.38144063949585, 3.690720319747925, 3.690720319747925, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 18.453601598739624, 11.072160959243774, 3.690720319747925, 18.453601598739624, 7.38144063949585, 3.690720319747925, 3.690720319747925, 11.072160959243774, 14.7628812789917, 14.7628812789917, 11.072160959243774, 3.690720319747925, 7.38144063949585, 7.38144063949585, 7.38144063949585, 11.072160959243774, 22.14432191848755, 3.690720319747925, 7.38144063949585, 11.072160959243774, 11.072160959243774, 11.072160959243774, 18.453601598739624, 14.7628812789917, 14.7628812789917, 3.690720319747925, 7.38144063949585, 22.14432191848755, 29.5257625579834, 11.072160959243774, 22.14432191848755, 14.7628812789917, 11.072160959243774, 11.072160959243774, 11.072160959243774, 132.8659315109253, 121.79377055168152, 73.8144063949585, 47.97936415672302, 14.7628812789917, 14.7628812789917, 7.38144063949585, 239.8968207836151, 265.7318630218506, 166.08241438865662, 158.70097374916077, 40.59792351722717, 14.7628812789917, 7.38144063949585]
gt = [0.0, 7.38144063949585, 14.7628812789917, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 0.0, 14.7628812789917, 0.0, 0.0, 7.38144063949585, 0.0, 14.7628812789917, 14.7628812789917, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 0.0, 7.38144063949585, 22.14432191848755, 14.7628812789917, 0.0, 0.0, 0.0, 14.7628812789917, 14.7628812789917, 14.7628812789917, 14.7628812789917, 14.7628812789917, 0.0, 0.0, 0.0, 0.0, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 7.38144063949585, 0.0, 14.7628812789917, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 0.0, 0.0, 29.5257625579834, 7.38144063949585, 14.7628812789917, 7.38144063949585, 7.38144063949585, 0.0, 14.7628812789917, 22.14432191848755, 14.7628812789917, 0.0, 0.0, 7.38144063949585, 7.38144063949585, 0.0, 22.14432191848755, 0.0, 22.14432191848755, 7.38144063949585, 22.14432191848755, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 0.0, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 7.38144063949585, 0.0, 22.14432191848755, 29.5257625579834, 14.7628812789917, 7.38144063949585, 0.0, 14.7628812789917, 0.0, 14.7628812789917, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 14.7628812789917, 14.7628812789917, 22.14432191848755, 14.7628812789917, 14.7628812789917, 22.14432191848755, 7.38144063949585, 7.38144063949585, 0.0, 0.0, 14.7628812789917, 29.5257625579834, 14.7628812789917, 36.90720319747925, 22.14432191848755, 7.38144063949585, 14.7628812789917, 22.14432191848755, 22.14432191848755, 22.14432191848755, 7.38144063949585, 7.38144063949585, 14.7628812789917, 7.38144063949585, 14.7628812789917, 14.7628812789917, 7.38144063949585, 147.628812789917, 118.1030502319336, 125.48449087142944, 118.1030502319336, 66.43296575546265, 81.19584703445435, 59.0515251159668, 36.90720319747925, 14.7628812789917, 14.7628812789917, 7.38144063949585, 22.14432191848755, 14.7628812789917, 0.0, 243.58754110336304, 236.2061004638672, 273.11330366134644, 258.35042238235474, 162.3916940689087, 169.77313470840454, 184.53601598739624, 132.8659315109253, 66.43296575546265, 14.7628812789917, 22.14432191848755, 7.38144063949585, 7.38144063949585, 7.38144063949585]

inp = np.reshape(inp,(14,7))
gt = np.reshape(gt,(14,14))

plt.figure()
plt.subplot(1,2,1)
plt.imshow(inp,cmap='gray')
plt.xlabel("input")
plt.subplot(1,2,2)
plt.imshow(gt,cmap='gray')
plt.xlabel("gt")

plt.show()