# ATMOxide
SI for `Predicting Dimensionality in Amine Templated Metal Oxides`.

All files mentioned in the main text can be found in [mentioned_in_text](./mentioned_in_text) folder.

## API
1. [AnalysisModule](./AnalysisModule) contains tools for:
    1. [Calculating](./AnalysisModule/calculator) structural and molecular descriptors.
    2. [Preparing](./AnalysisModule/prepare) data entries, e.g. disorder cleanup, organic-inorganic split. It also contains a 
    3. [Routine](./AnalysisModule/routines) functions supporting 1. and 2.

    These tools were used in [DataGeneration](DataGeneration).

2. [MLModule](./MLModule) contains a series of classes/functions used in [DimPredict](DimPredict) to predict dimensionality.

## Figures
Folder [Floats](./Floats) contains all codes/data used to generate figures. Note the numbering may differ from what is present in main text/SI.