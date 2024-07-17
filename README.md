# README

PEPITA-tools contains utilities relevant to the quantification of fluorescent intensity values from fluorescence microscopy images:
- `pipeline.py` and related python scripts, which take such images as input and produce numeric values and plots, and
- ImageJ macro scripts contained in `ImageJ_scripts/`, which help with the creation of mask files, an optional step in the quantification workflow.

## Usage

1. Clone this repository
	- Currently this code is not packaged as a proper library, and as such must be downloaded in repository form to properly run.
2. Install python if not already installed
	- See https://www.python.org/downloads/ for more information
3. Ensure script dependencies are installed with `requirements.txt`, like
	```
	python -m pip install -r requirements.txt
	```
4. Set the `log_dir` setting in the `config-ext.ini` file
	- Create a new `config-ext.ini` file in the repository directory if one doesn't yet exist.
	- Set this setting to a location where you have write privileges and where logging information can be conveniently written. If the supplied path does not exist, a new directory will be created.
5. Prepare image data
	- Brightfield and fluorescent microscopy images should be saved in a location accessible to this script
	- If particular images should be excluded, or if any require custom masking, that should be done by creating mask files at this time (see _Custom masking_ below).
6. Prepare CSV plate template
	- A plate template must be supplied to provide meaningful labels to the images being analyzed by the system. The pipeline will apply labels to images by matching them up in order from the top left, to the right then down.
	- An example experiment of a single 96-well plate with untreated controls and increasing doses of drug ABC might look like the following, which would label the first through tenth images as _ABC 0μM_, the 11th through 20th images as _ABC 1μM_, and so on, down to the 51st through 60th images as _ABC 16μM_.

|  |  |  |  |  |  |  |  |  |  |  |  |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|  | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM | ABC 0μM |  |
|  | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM | ABC 1μM |  |
|  | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM | ABC 2μM |  |
|  | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM | ABC 4μM |  |
|  | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM | ABC 8μM |  |
|  | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM | ABC 16μM |  |
|  |  |  |  |  |  |  |  |  |  |  |  |

7. Run the analysis pipeline with appropriate runtime arguments
	- See _Runtime arguments_ below for more information.

Alternatively, this pipeline can be run as a notebook in the cloud, e.g., through Google Colab (see _Notebook Execution_ section below). In this case, steps 1-5 are to be done in the context of the notebook's execution environment; see the notebook itself for more information.

## `pipeline.py` and Related Scripts

This utility quantifies fluorescence data via roughly the following steps:
1. Identifies and masks fish larva in each image, to exclude noise present in the image away from the immediate proximity of the fish,
1. Adjusts for autofluorescence using a non-signal-containing channel,
1. Identifies and masks the top 15 brightest fluorescent puncta, then discards the top 5,
1. Scores each image based on the sum of the pixel values of the masked image,
1. Standardizes the image scores based on provided control condition(s) for each plate,
1. Fits dose response curves for single treatments and/or interaction parameters for combo treatments.

Each step is optional (except for scoring) and is included or excluded based on the configured parameters of each execution, as explained below.

Further information on each of the python scripts can be found in their help text, which can be retrieved by executing `python [script] --help` on a command line with access to python.

### Configuration

Configuration is achieved by two main routes: the configuration file and runtime arguments. The exception to this is exclusion or custom masking of individual supplied images, which can be achieved by supplying separate mask files as needed.

#### Configuration file

A default configuration file can be found at [config.ini](/config.ini). This file defines both the available configuration settings and their default values.

To override default values, create a file next to the default configuration file, called `config-ext.ini`, and set the desired values within.

The configuration settings take the following forms:
- `absolute*`: Positive integer. Indicates "absolute" (that is, not scaled relative to plate controls) fluorescent unit values that should be considered roughly the maximum and minimum expected values for the relevant type of experiment. These values are used to calculate "absolute" percentage fluorescence in certain output plots. `*_infection` values are used in `infection.py`; `*_ototox` values are used in `pipeline.py`.
- `channel*`: Positive integer. Indicates the index of the desired "channel" (or color) within the supplied images. `channel_main_*` indices are used to retrieve fluorescence intensity data; `channel_subtr_*` indices are used to retrieve the aforementioned non-signal-containing data used to adjust for autofluorescence. `*_infection` and `*ototox` are as above.
- `filename_replacement_*`: String. A delimiter (substring) is provided in the `filename_replacement_delimiter` setting; all other configurations are 2-value replacements delimited by the substring provided. `*_brightfield_*` provides a replacement the system will use to get from the supplied filenames (fluorescence images) to the associated brightfield images; `*_mask_*`, from the supplied images to any associated custom masks (see below); `*_subtr_*`, from the supplied images to any associated non-signal-containing, autofluorescence-canceling, images. `*_infection` and `*ototox` are as above.
- `log_dir`: String. Indicates the absolute path of a directory the scripts may use to output logging information (the quantity of which will be determined by runtime arguments).

A `log_dir` should always be provided. All other configuration settings may be fine to leave as default.

#### Runtime arguments

The pipeline may be executed either as a CLI (like `python pipeline.py [arguments]`) or as a library (e.g., from with a Jupyter notebook). The script takes the same arguments either way.

The CLI help text can provide up-to-date information on which arguments can be supplied, what form they should take, and whether they are required. At the time of this writing, this text looks like the following (though no guarantees this will remain in sync with the most up-to-date version):

```
usage: pipeline.py [-h] [-cb] [-cv [CONVERSIONS ...]]
                   [-ppc [PLATE_POSITIVE_CONTROL ...]]
                   [--plate-info PLATE_INFO] [-tp TREATMENT_PLATEFILE]
                   [--absolute-chart] [--talk] [-ch CHARTFILE] [-p PLATEFILE]
                   [-pc [PLATE_CONTROL ...]] [-pi [PLATE_IGNORE ...]]
                   [-g GROUP_REGEX] [-c CAP] [-d] [-s]
                   imagefiles [imagefiles ...]

Analyzer for images of whole zebrafish with fluorescent neuromasts, for the
purposes of measuring hair cell damage under drug-combination conditions.
Reports values relative to control.

positional arguments:
  imagefiles            The absolute or relative filenames where the relevant
                        images can be found.

options:
  -h, --help            show this help message and exit
  -cb, --checkerboard   If present, the input will be treated as a
                        checkerboard assay, with output produced accordingly.
  -cv [CONVERSIONS ...], --conversions [CONVERSIONS ...]
                        List of conversions between dose concentration labels
                        and concrete values, each as a separate argument, each
                        delimited by an equals sign. For instance, ABC50 might
                        be an abbreviation for the EC50 of drug ABC, in which
                        case the concrete concentration can be supplied like
                        "ABC50=ABC 1mM" (make sure to quote, or escape
                        spaces).
  -ppc [PLATE_POSITIVE_CONTROL ...], --plate-positive-control [PLATE_POSITIVE_CONTROL ...]
                        Labels to treat as the positive control conditions in
                        the plate schematic (i.e. conditions showing maximum
                        effect). These wells are used to normalize all values
                        in the plate for more interpretable results. Any
                        number of values may be passed.
  --plate-info PLATE_INFO
                        Any information identifying the plate(s) being
                        analyzed that should be passed along to files created
                        by this process.
  -tp TREATMENT_PLATEFILE, --treatment-platefile TREATMENT_PLATEFILE
                        CSV file containing a schematic of the plate in which
                        the imaged fish were treated. Used to chart responses
                        by treatment location, if desired. Row and column
                        headers are optional. The cell values are essentially
                        just arbitrary labels: results will be grouped and
                        charted according to the supplied values.
  --absolute-chart      If present, a plate graphic will be generated with
                        absolute (rather than relative) brightness values.
  --talk                If present, images will be generated with the Seaborn
                        "talk" context.
  -ch CHARTFILE, --chartfile CHARTFILE
                        If supplied, the resulting numbers will be charted at
                        the given filename.
  -p PLATEFILE, --platefile PLATEFILE
                        CSV file containing a schematic of the plate from
                        which the given images were taken. Row and column
                        headers are optional. The cell values are essentially
                        just arbitrary labels: results will be grouped and
                        charted according to the supplied values.
  -pc [PLATE_CONTROL ...], --plate-control [PLATE_CONTROL ...]
                        Labels to treat as the control condition in the plate
                        schematic. These wells are used to normalize all
                        values in the plate for more interpretable results.
                        Any number of values may be passed.
  -pi [PLATE_IGNORE ...], --plate-ignore [PLATE_IGNORE ...]
                        Labels to ignore (treat as null/empty) in the plate
                        schematic. Empty cells will automatically be ignored,
                        but any other null values (e.g. "[empty]") must be
                        specified here. Any number of values may be passed.
  -g GROUP_REGEX, --group-regex GROUP_REGEX
                        Pattern to be used to match group names that should be
                        included in the results. Matched groups will be
                        included, groups that don't match will be ignored.
                        Control wells will always be included regardless of
                        whether they match.
  -c CAP, --cap CAP     Exclude well values larger than the given integer,
                        expressed as a percentage of the median control value.
  -d, --debug           Indicates intermediate processing images should be
                        output for troubleshooting purposes. Including this
                        argument once will yield one intermediate image per
                        input file, twice will yield several intermediate
                        images per input file.
  -s, --silent          If present, printed output will be suppressed. More
                        convenient for programmatic execution.
```

If executing as a library, long-form arguments should be supplied, with any dashes replaced by underscores.

#### Custom masking

In the base case, the PEPITA-tools pipeline will analyze each brightfield image, attempting to locate a fish larva and subsequently draw a mask around it. This can be prevented by supplying a mask for any given image before pipeline execution, in which case the supplied mask will be used instead of a new one generated.

Custom masks are commonly necessary in any of the following situations:
- An image should be excluded from analysis.
- A larva is partly off the image, or extremely out of plane, such that the object detection algorithm fails to identify it.
- A fluorescent strain of zebrafish is being used, necessitating different masking requirements (e.g., to exclude a fluorescent inner ear).

Custom masks should conform to the following specification:
- 8-bit image file
- Of the same format as the microscopy images
- Having the same resolution as the microscopy images
- Single channel
- Areas to exclude are black (pixel value is 0); areas where signal should be processed are white (pixel value is 255)

The provided ImageJ scripts can be used to produce these images more easily (see below).

### Command-line Execution

Run `python pipeline.py --help` for info on how to execute the pipeline via command line.

### Notebook Execution

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ma-lab-cgidr/PEPITA-tools/blob/master/interactive_pipeline.ipynb)

## ImageJ Scripts

Four scripts are provided to assist in the production of custom mask files:
- `OpenForMasking.ijm`: Prompts for a directory and a substring (that should match the signal-containing fluorescence files), then opens all image files within the given directory that contain the given substring and merges them with their associated brightfield file. This allows for easy mask production, by using the ImageJ "Freehand selections" tool to draw the mask shape around the observed fish (with both channels visible). `SaveFishMask.ijm` can then be used to save the selection as a mask in the proper format.
- `OpenForMaskingSingle.ijm`: Same as above, but prompts for a single image file and only opens and prepares the one image.
- `SaveFishMask.ijm`: Takes a selection of an image opened with either `OpenForMasking.ijm` or `OpenForMaskingSingle.ijm` and makes the necessary operations to save it as a mask in the location expected by the analysis pipeline.
- `SaveFishNullMask.ijm`: Makes a "null" mask (i.e. a mask that excludes the whole image) from an image opened with either `OpenForMasking.ijm` or `OpenForMaskingSingle.ijm`. This is the easiest way to exclude a particular image from analysis.

For installation and explanation, see:
- https://imagej.net/ij/developer/macro/macros.html
- https://imagej.net/scripting/macro#installing-macros
- [Macro installer script](/ImageJ_scripts/macroize.sh)
