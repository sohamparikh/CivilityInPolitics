The civility data is located in nlpgrid at `/nlp/data/corpora/civility_data/`.
The shows from the month of February are in the `FebShows` directory and the PBS shows from March, April and May are in the `PBS_MarchAprilMay2019` directory.
The videos for the minute long snippets are in `Civil - Uncivil project.zip`, the transcriptions are in `minute_transcriptions`. The mapping from the videos to the transcriptions is provided in the `mapping_minute_transcriptions.json` file.

## Getting Toxicity Scores

You can find the functions to get the toxicity scores in `analysis_utils.py`. Please go through the Politeness API documentation about how to use Python to call it. You can use the provided API keys or you can use your own API keys.

## Getting Politeness Scores

Getting the politeness scores is a bit of a task. You first need to run the `parser_politeness.py` script on your data to get the parse for your sentences. Once the sentences are parsed, save them along with the parses in a file. Then, you need to run the `politeness/compute_politeness.py` script on these parses. Note that the latter only works with Python 2. 
