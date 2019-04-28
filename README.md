# wiki-download
Script to download, tokenize and numericalize the wikipedia. Based on the script by Sebastian Ruder


## Running the script

* Run the container:
  ```docker run -it --runtime=nvidia -v $(pwd):/data lucasarb/deeplearning bash```
* Go to the right folder on the container:
  ```cd /data/scripts```
* Change the `prepare_wiki.sh` script to be executable by the user:
  ```chmod +x prepare_wiki.sh```
* Run the script:
  ```./prepare_wiki.sh```
  
## Training the language model(lm-training.py)

* At the moment there are still some problems on this. If you want to try it feel free to modify it. To run use:
  ```python lm-training.py```
  
