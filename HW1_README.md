# Running Instructions

------------------ HW1 PART 1 --------------------------

For HW1.py file:
Enter dataset name (Line 7):
Default is dip-hard-eff.csv
dip-har-eff.csv | drift-har-eff.csv | set-har-eff.csv

Enter batchSize (Line 87):
1 | 5 | 10 | 15 | 20

* Note: dip-har-eff.csv does not have enough values to use batch size 20 gradient descent *

Enter number of epochs to run for (Line 88):
Default is 100 epochs

You can change learning rate as well on Line 90

* Also note: On Lines 96-97 you can change the labels for your graphs to plot the regressions *

------------------ HW1 PART 2 --------------------------

For HW1_albacore.py file:
Enter which dataset columns to use, select any combination of two:
 - dataset_df[:,1] : Cadmium
 - dataset_df[:,2] : Mercury
 - dataset_df[:,3] : Lead

Enter batchSize (Line 87):
1 | 5 | 10 | 15 | 20

Enter number of epochs to run for (Line 88):
Default is 100 epochs

You can change learning rate as well on Line 90

* Also note: On Lines 96-97 you can change the labels for your graphs to plot the regressions *
- Default is Cadmium and Lead
