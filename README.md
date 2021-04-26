# TMAE: Unsupervised Patient Representation Learning using Claims Data for Risk Stratification and Analysis

This repository contains the tensorflow implementation of the following paper:

Paper Name: to be updated

Authors: to be updated

Abstract: to be updated

Paper Link: to be updated

# Environment
Ubuntu16.04, Python3.7, TensorFlow2.1


## Stratification results on three disease-specific populations

|      Depression cohort      | cluster-1 n=481 | Cluster-2 n=796 |
|:---------------------------:|:---------------:|:---------------:|
|           Aver. Age         |        11       |        13       |
|           Female %          |        29       |        64       |
|       Aver. # OP visit      |        8.2      |       10.2      |
|       Aver. # IP visit      |       0.21      |       0.06      |
|   Median RX cost (Y1 / Y2)  |   $1861/ $1662  |    $164/ $149   |
| Median Total cost (Y1 / Y2) | $2650/ $2574    |   $1508/ $1124  |

|       Autism cohort           |     Cluster-1     n=150    |     Cluster-2     n=133    |     Cluster-3     n=226    |
|:-----------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|
|               Age             |              10            |              6             |              10            |
|            Female %           |              19            |              20            |              18            |
|           # OP visit          |             9.3            |             12.2           |             8.1            |
|           # IP visit          |             0.02           |            0.008           |             0.05           |
|       RX cost     (Y1/Y2)     |        $86 /     $115      |        $38 /     $32       |      $1139 /     $1168     |
|     Total cost     (Y1/Y2)    |       $1355 /     $884     |      $1221 /     $1070     |      $2087 /     $1972     |

|       Diabetes cohort       |     Cluster-1     n=481    |     Cluster-2     n=796    |     Cluster-1     n=481    |     Cluster-2     n=796    |
|:---------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|:--------------------------:|
|           Aver. Age         |              11            |              13            |              11            |              13            |
|           Female %          |              29            |              64            |              29            |              64            |
|       Aver. # OP visit      |             8.2            |             10.2           |             8.2            |             10.2           |
|       Aver. # IP visit      |             0.21           |             0.06           |             0.21           |             0.06           |
|   Median RX cost (Y1 / Y2)  |      $ 1861/     $ 1662    |       $ 164/     $ 149     |      $ 1861/     $ 1662    |        $164/     $149      |
| Median Total cost (Y1 / Y2) |      $ 2650/     $ 2574    |      $ 1508/     $ 1124    |      $ 2650/     $ 2574    |       $1508/     $1124     |


