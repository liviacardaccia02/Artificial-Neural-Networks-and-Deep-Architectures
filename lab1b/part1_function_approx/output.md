# Output

Dataset information
-------------------
x range: -5.0 to 4.5, step=0.5
y range: -5.0 to 4.5, step=0.5
Grid size: 20 x 20
Number of samples: 400
patterns shape: (2, 400)
targets shape: (1, 400)
Target range: [-0.493262, 0.500000]

BASELINE APPROXIMATION RESULTS:

BASELINE CONFIGURATION
---------------------
Architecture: 2-20-1
eta=0.2, alpha=0.9, epochs=1000, seed=42
Final all-data performance: MSE=0.00010904, RMSE=0.01044238, MAE=0.00850347, MaxAbsError=0.03368805
Initial training MSE: 0.20831094
Final training MSE:   0.00010920
Minimum training MSE: 0.00010920
Epoch of minimum MSE: 999
Epoch with MSE < 0.01: 77


Hidden Layer Size Studies Results:

  Hidden      Final MSE        Min MSE     Conv.<0.01      MaxAbsErr
----------------------------------------------------------------------
       1     0.06252871     0.06253445              -     0.66695162
       2     0.03791869     0.03791870              -     0.47572288
       3     0.00213734     0.00213765            331     0.14658859
       4     0.00180165     0.00180194            174     0.12673284
       5     0.00030806     0.00030822            192     0.06515217
       7     0.00026887     0.00026904            123     0.07253285
      10     0.00018714     0.00018733            108     0.04825719
      15     0.00028396     0.00028412             91     0.04811397
      20     0.00010904     0.00010920             77     0.03368805
      25     0.00006171     0.00006177             50     0.04578395


Representative hidden-layer sizes:
Hidden=1: MSE=0.06252871, RMSE=0.25005742, MAE=0.20257415, MaxAbsError=0.66695162
Hidden=3: MSE=0.00213734, RMSE=0.04623132, MAE=0.03410719, MaxAbsError=0.14658859
Hidden=5: MSE=0.00030806, RMSE=0.01755174, MAE=0.01300468, MaxAbsError=0.06515217
Hidden=10: MSE=0.00018714, RMSE=0.01367979, MAE=0.01096616, MaxAbsError=0.04825719
Hidden=20: MSE=0.00010904, RMSE=0.01044238, MAE=0.00850347, MaxAbsError=0.03368805
Hidden=25: MSE=0.00006171, RMSE=0.00785577, MAE=0.00583538, MaxAbsError=0.04578395


MODEL COMPARISON
----------------
Candidate selected for subsequent experiments: 2-25-1
Final all-data MSE: 0.00006171
RMSE: 0.00785577
MAE: 0.00583538
Maximum absolute error: 0.04578395
Minimum training MSE during run: 0.00006177
Convergence epoch (MSE < 0.01): 50


GENERALISATION STUDY RESULTS:

 Train %  N train    N val      Train MSE        Val MSE        All MSE     Conv.<0.01
----------------------------------------------------------------------------------------------------
     80%      320       80     0.00005693     0.00006469     0.00005848             50
     60%      240      160     0.00006086     0.00006419     0.00006220             50
     40%      160      240     0.00007283     0.00009431     0.00008572             49
     20%       80      320     0.00008388     0.00026392     0.00022791             48


CONVERGENCE STUDY RESULTS:

     eta    alpha      Final MSE        Min MSE     Conv.<0.01     Stable
-------------------------------------------------------------------------------------
   0.010     0.00     0.04442744     0.04443721              -       True
   0.010     0.50     0.03599229     0.03600693              -       True
   0.010     0.90     0.00749353     0.00750758            852       True
   0.050     0.00     0.01997826     0.01999793              -       True
   0.050     0.50     0.00743343     0.00744723            846       True
   0.050     0.90     0.00041883     0.00041944            176       True
   0.100     0.00     0.00742601     0.00743978            846       True
   0.100     0.50     0.00200919     0.00201295            424       True
   0.100     0.90     0.00015344     0.00015368             91       True
   0.200     0.00     0.00200847     0.00201223            423       True
   0.200     0.50     0.00057810     0.00057901            212       True
   0.200     0.90     0.00006171     0.00006177             50       True
   0.300     0.00     0.00093998     0.00094169            282       True
   0.300     0.50     0.00031921     0.00031967            142       True
   0.300     0.90     0.00004839     0.00004841             36       True

CONVERGENCE COMPARISON
----------------------
     eta    alpha    Conv. epoch      Final MSE
-------------------------------------------------------
   0.300     0.90             36     0.00004839
   0.200     0.90             50     0.00006171
   0.100     0.90             91     0.00015344
   0.300     0.50            142     0.00031921
   0.050     0.90            176     0.00041883
   0.200     0.50            212     0.00057810
   0.300     0.00            282     0.00093998
   0.200     0.00            423     0.00200847
   0.100     0.50            424     0.00200919
   0.050     0.50            846     0.00743343
   0.100     0.00            846     0.00742601
   0.010     0.90            852     0.00749353

Fastest configuration reaching MSE < 0.01:
eta=0.3, alpha=0.9
Convergence epoch=36
Final all-data MSE=0.00004839
Baseline all-data MSE=0.00010904
Baseline convergence epoch=77


REPEATED RANDOM INITIALISATIONS
-------------------------------
seed=0: final all-data MSE=0.00019381
seed=1: final all-data MSE=0.00018081
seed=2: final all-data MSE=0.00008654
seed=3: final all-data MSE=0.00008861
seed=4: final all-data MSE=0.00011093
Mean final MSE: 0.00013214
Std final MSE:  0.00004603
Variance final MSE: 0.00000000
Convergence epoch mean/std: 70.60 / 16.73


NOISE STUDY RESULTS:

    Variance    Train noisy MSE      All clean MSE   Validation clean MSE
---------------------------------------------------------------------------
      0.0001         0.00013596         0.00006119             0.00006297
      0.0010         0.00087429         0.00010144             0.00008816
      0.0100         0.00835941         0.00052949             0.00042562
      0.0500         0.04150215         0.00282957             0.00244303
      0.1000         0.08302513         0.00536664             0.00466047


==========================================================================================
SECTION 3.1.3 — REPORT-READY SUMMARY
==========================================================================================

DATASET
Grid: 20x20 = 400 samples
Function: f(x,y) = exp(-(x^2+y^2)/10) - 0.5

BASELINE
Architecture: 2-20-1
eta=0.2, alpha=0.9, epochs=1000
All-data: MSE=0.00010904, RMSE=0.01044238, MAE=0.00850347, MaxAbsError=0.03368805
Convergence epoch (<0.01): 77

HIDDEN-LAYER STUDY
hidden= 1 | final MSE=0.06252871 | min MSE=0.06253445 | conv=-
hidden= 2 | final MSE=0.03791869 | min MSE=0.03791870 | conv=-
hidden= 3 | final MSE=0.00213734 | min MSE=0.00213765 | conv=331
hidden= 4 | final MSE=0.00180165 | min MSE=0.00180194 | conv=174
hidden= 5 | final MSE=0.00030806 | min MSE=0.00030822 | conv=192
hidden= 7 | final MSE=0.00026887 | min MSE=0.00026904 | conv=123
hidden=10 | final MSE=0.00018714 | min MSE=0.00018733 | conv=108
hidden=15 | final MSE=0.00028396 | min MSE=0.00028412 | conv=91
hidden=20 | final MSE=0.00010904 | min MSE=0.00010920 | conv=77
hidden=25 | final MSE=0.00006171 | min MSE=0.00006177 | conv=50

SELECTED CANDIDATE: 2-25-1

GENERALISATION
 80% train (320 samples) | train MSE=0.00005693 | val MSE=0.00006469 | all-data MSE=0.00005848 | conv=50
 60% train (240 samples) | train MSE=0.00006086 | val MSE=0.00006419 | all-data MSE=0.00006220 | conv=50
 40% train (160 samples) | train MSE=0.00007283 | val MSE=0.00009431 | all-data MSE=0.00008572 | conv=49
 20% train ( 80 samples) | train MSE=0.00008388 | val MSE=0.00026392 | all-data MSE=0.00022791 | conv=48

CONVERGENCE / HYPERPARAMETERS
eta=0.010, alpha=0.00 | final MSE=0.04442744 | conv=- | stable=True
eta=0.010, alpha=0.50 | final MSE=0.03599229 | conv=- | stable=True
eta=0.010, alpha=0.90 | final MSE=0.00749353 | conv=852 | stable=True
eta=0.050, alpha=0.00 | final MSE=0.01997826 | conv=- | stable=True
eta=0.050, alpha=0.50 | final MSE=0.00743343 | conv=846 | stable=True
eta=0.050, alpha=0.90 | final MSE=0.00041883 | conv=176 | stable=True
eta=0.100, alpha=0.00 | final MSE=0.00742601 | conv=846 | stable=True
eta=0.100, alpha=0.50 | final MSE=0.00200919 | conv=424 | stable=True
eta=0.100, alpha=0.90 | final MSE=0.00015344 | conv=91 | stable=True
eta=0.200, alpha=0.00 | final MSE=0.00200847 | conv=423 | stable=True
eta=0.200, alpha=0.50 | final MSE=0.00057810 | conv=212 | stable=True
eta=0.200, alpha=0.90 | final MSE=0.00006171 | conv=50 | stable=True
eta=0.300, alpha=0.00 | final MSE=0.00093998 | conv=282 | stable=True
eta=0.300, alpha=0.50 | final MSE=0.00031921 | conv=142 | stable=True
eta=0.300, alpha=0.90 | final MSE=0.00004839 | conv=36 | stable=True

STOCHASTICITY
Seeds: [0, 1, 2, 3, 4]
Mean MSE: 0.00013214
Std MSE:  0.00004603
Variance:  0.00000000

OPTIONAL NOISE
variance=0.0001 | noisy train MSE=0.00013596 | all clean MSE=0.00006119 | val clean MSE=0.00006297
variance=0.0010 | noisy train MSE=0.00087429 | all clean MSE=0.00010144 | val clean MSE=0.00008816
variance=0.0100 | noisy train MSE=0.00835941 | all clean MSE=0.00052949 | val clean MSE=0.00042562
variance=0.0500 | noisy train MSE=0.04150215 | all clean MSE=0.00282957 | val clean MSE=0.00244303
variance=0.1000 | noisy train MSE=0.08302513 | all clean MSE=0.00536664 | val clean MSE=0.00466047
==========================================================================================
END OF EXPERIMENTS
==========================================================================================
