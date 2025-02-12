# # -------------------------------------------------------------------------  
#  * This program is based on a one-pass algorithm for the calculation of an  
#  * array of autocorrelations r[1], r[2], ... r[K].  The key feature of this 
#  * algorithm is the circular array 'hold' which stores the (K + 1) most 
#  * recent data points and the associated index 'p' which points to the 
#  * (rotating) head of the array. 
#  * 
#  * Data is read from a text file in the format 1-data-point-per-line (with 
#  * no blank lines).  Similar to programs UVS and BVS, this program is
#  * designed to be used with OS redirection. 
#  * 
#  * NOTE: the constant K (maximum lag) MUST be smaller than the # of data 
#  * points in the text file, n.  Moreover, if the autocorrelations are to be 
#  * statistically meaningful, K should be MUCH smaller than n. 
#  *
#  * Name              : acs.c  (AutoCorrelation Statistics) 
#  * Author            : Steve Park & Dave Geyer 
#  * Language          : ANSI C
#  * Latest Revision   : 2-10-97 
#  * Compile with      : gcc -lm acs.c
#  * Execute with      : acs.out < acs.dat
#  # Translated by     : Philip Steele 
#  # Language          : Python 3.3
#  # Latest Revision   : 3/26/14
#  * Execute with      : python acs.py < acs.dat
#  * ------------------------------------------------------------------------- 
#  */
"""
#include <stdio.h>
#include <math.h>

import sys
from math import sqrt

K = 50                             # K is the maximum lag */
SIZE = (K + 1)


i = 0                          # data point index              */
sum = 0.0                      # sums x[i]                     */
j = 0                          # lag index                     */
hold = []                      # K + 1 most recent data points */
p = 0                          # points to the head of 'hold'  */
cosum = [0 for i in range(0,SIZE)]    # cosum[j] sums x[i] * x[i+j]   */


while (i < SIZE):              # initialize the hold array with */
  x = float(sys.stdin.readline())     # the first K + 1 data values    */
  sum += x
  hold.append(x)
  i += 1
#EndWhile

x = sys.stdin.readline()   

while (x):
  for j in range(0,SIZE):
    cosum[j] += hold[p] * hold[(p + j) % SIZE]
  x = float(x) #lines read in as string
  sum    += x
  hold[p] = x
  p       = (p + 1) % SIZE
  i += 1 
  x = sys.stdin.readline()
#EndWhile
n = i #the total number of data points

while (i < n + SIZE):         # empty the circular array */
  for j in range(0,SIZE):
    cosum[j] += hold[p] * hold[(p + j) % SIZE]
  hold[p] = 0.0
  p       = (p + 1) % SIZE
  i += 1 
#EndWhile

mean = sum / n
for j in range(0,K+1):  
  cosum[j] = (cosum[j] / (n - j)) - (mean * mean)

print("for {0} data points".format(n))
print("the mean is ... {0:8.2f}".format(mean))
print("the stdev is .. {0:8.2f}\n".format(sqrt(cosum[0])))
print("  j (lag)   r[j] (autocorrelation)\n")
for j in range(1,SIZE):
  print("{0:3d}  {1:11.3f}".format(j, cosum[j] / cosum[0]))


# C output:
# for 500 data points
# the mean is ...     5.68
# the stdev is ..     4.72

#   j (lag)   r[j] (autocorrelation)
#   1        0.966
#   2        0.931
#   3        0.898
#   4        0.862
#   5        0.831
#   6        0.799
#   7        0.764
#   8        0.730
#   9        0.700
#  10        0.668
#  11        0.641
#  12        0.612
#  13        0.583
#  14        0.554
#  15        0.524
#  16        0.494
#  17        0.465
#  18        0.437
#  19        0.410
#  20        0.382
#  21        0.358
#  22        0.334
#  23        0.308
#  24        0.277
#  25        0.247
#  26        0.221
#  27        0.193
#  28        0.168
#  29        0.148
#  30        0.131
#  31        0.119
#  32        0.105
#  33        0.089
#  34        0.076
#  35        0.069
#  36        0.064
#  37        0.062
#  38        0.062
#  39        0.061
#  40        0.056
#  41        0.054
#  42        0.060
#  43        0.062
#  44        0.067
#  45        0.068
#  46        0.059
#  47        0.055
#  48        0.053
#  49        0.052
#  50        0.050
"""
from math import sqrt

K = 1  # K is the maximum lag
SIZE = (K + 1)

def acsold(average):
    i = 0
    sum = 0.0
    j = 0
    hold = []
    p = 0
    cosum = [0 for _ in range(SIZE)]

    while i < SIZE and i < len(average):
        x = float(average[i])
        sum += x
        hold.append(x)
        i += 1

    while i < len(average):
        for j in range(SIZE):
            cosum[j] += hold[p] * hold[(p + j) % SIZE]
        x = float(average[i])
        sum += x
        hold[p] = x
        p = (p + 1) % SIZE
        i += 1

    n = i

    while i < n + SIZE:
        for j in range(SIZE):
            cosum[j] += hold[p] * hold[(p + j) % SIZE]
        hold[p] = 0.0
        p = (p + 1) % SIZE
        i += 1

    mean = sum / n
    # Calculate variance (cosum[0] is already the sum of x[i] * x[i])
    var = cosum[0] / n - (mean * mean)
    stdev = sqrt(var)

    for j in range(K + 1):
        cosum[j] = (cosum[j] / (n - j)) - (mean * mean)

    # Normalize autocorrelation by dividing by variance
    print("for {0} data points".format(n))
    print("the mean is ... {0:8.2f}".format(mean))
    print("the stdev is .. {0:8.2f}\n".format(stdev))
    print("  j (lag)   r[j] (autocorrelation)\n")
    for j in range(1, SIZE):
        normalized_autocorr = cosum[j] / var  # Normalize by variance
        print("{0:3d}  {1:11.3f}".format(j, normalized_autocorr))

    return cosum[1] / cosum[0]

from math import sqrt

def acs(data):
    K = 1  # K is the maximum lag
    SIZE = (K + 1)

    i = 0                          # data point index
    sum = 0.0                       # sums x[i]
    j = 0                           # lag index
    hold = []                       # K + 1 most recent data points
    p = 0                           # points to the head of 'hold'
    cosum = [0 for i in range(0, SIZE)]  # cosum[j] sums x[i] * x[i+j]

    # Initialize the hold array with the first K + 1 data values
    while i < SIZE:
        x = float(data[i])  # Read the first K + 1 data values
        sum += x
        hold.append(x)
        i += 1

    # Process remaining data points
    for x in data[i:]:
        for j in range(0, SIZE):
            cosum[j] += hold[p] * hold[(p + j) % SIZE]
        x = float(x)  # Read next value from list
        sum += x
        hold[p] = x
        p = (p + 1) % SIZE
        i += 1

    n = i  # Total number of data points

    # Empty the circular array
    for _ in range(SIZE):
        for j in range(0, SIZE):
            cosum[j] += hold[p] * hold[(p + j) % SIZE]
        hold[p] = 0.0
        p = (p + 1) % SIZE
        i += 1

    mean = sum / n
    for j in range(0, K+1):
        cosum[j] = (cosum[j] / (n - j)) - (mean * mean)

    # Print results for lag 1 only
    print("For {0} data points".format(n))
    print("The mean is ... {0:8.2f}".format(mean))
    print("The stdev is .. {0:8.2f}\n".format(sqrt(cosum[0])))
    print("  j (lag)   r[j] (autocorrelation)\n")
    print("  1  {0:11.3f}".format(cosum[1] / cosum[0]))  # Only lag 1

    return cosum[1] / cosum[0]  # Return the autocorrelation at lag 1
