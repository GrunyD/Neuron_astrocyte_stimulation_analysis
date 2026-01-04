# Spatiotemporally distinctive astrocytic and neuronal responses to repetitive intracortical microstimulation
[bioRxiv](https://www.biorxiv.org/content/10.64898/2026.01.02.697363v1)
This repository is code used for image analyses for the article.

**Authors:** David Grundfest^1*^, Kayeon Kim^1*^, Alexandra Katherine Isis Yonza^1^, Jeremi Tadeusz Podsiadlo^1^, Lechan Tao^1^, Xiao Zhang^1^, Krzysztof Kucharz^1^, Yan Zhang^3^, Anpan Han^2^, Barbara Lind^1†^, Changsi Cai^1†^

**Affiliations**
1 Department of Neuroscience, Faculty of Health and Medical Science, University of Copenhagen, DK-2200, Copenhagen, Denmark 
2 Department of Civil and Mechanical Engineering, Technical University of Denmark, Lyngby, Denmark
3 Shanghai General Hospital affiliated with Shanghai Jiao Tong University School of Medicine, Shanghai, China

† Correspondence e-mail: 
Changsi Cai, 
ccai@sund.ku.dk

Barbara Lykke Lind
barbarall@sund.ku.dk



*The authors contributed equally.


## Abstract

Astrocytes are increasingly recognized as active modulators of neuronal synaptic transmission. Intracortical microstimulation (ICMS) is widely used to manipulate neuronal activity, yet the accompanying astrocytic responses remain poorly characterized. Using dual-color in vivo two-photon calcium imaging to simultaneously monitor neurons and astrocytes, we show that ICMS elicits astrocytic activation with spatiotemporal features that diverge from those of neurons. Astrocytes were recruited at stimulation intensities as low as 10μA, thresholds sufficient to activate neurons, indicating that astrocytes robustly sense electrical perturbation. Unlike neurons, however, astrocytic responses were spatially heterogeneous and temporally variable across trials. At higher stimulation intensities (>=50 μA), astrocytic responsiveness, i.e., response peak amplitude, and number of responsive trials, progressively attenuated across repeated trials, in contrast to the stable and consistent neuronal responses. Although neuronally driven, astrocytes exhibited a distinct response profile under the same stimulation parameter, revealing a unique component of electrically evoked cortical activity that underscores the importance of incorporating glial physiology into future neuroprosthetic strategies. 


## Structure of the code

All data were first converted into tiff from oir in script `convert_data_to_tiff.py`.
Then the singal signal processing (apply grid, low-pass filter, substract baseline) was done ahead of time in script `precompute.py`.
The notebooks contain all the statistical analyses and plotting that is shown in the article.
Other scripts (like `plotting.py`) only have functions to import into the notebooks.

Any questions regarding code can be communicated with owner of this repository.
