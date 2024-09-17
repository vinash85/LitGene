This work is on trail webrun with deployement on (https://www.avisahuai.com/tools) select LitGene tool. 
Deployment github (https://github.com/vinash85/GENELLM_WEBAPP) can be used for self deployment of the webpage. 
Use feedback page on the LitGene tool page or contact authors to provide feedback.
BioArxiv link: https://www.biorxiv.org/content/10.1101/2024.08.07.606674v1


Code Refactoring in progress and you can run sample code with instructions below:

Here is how to run sample code using Docker.
1. Build docker:
      a. Go to location of docker file after git clone "cd LitGene/dependencies/docker/"
      b. Build docker "docker build . -t litgene"
2. Run docker image/create container:
      c. "docker run --name Litgene --gpus=all --previleged --ports 8888:8888 -v litgene_location:/home/tailab/LitGene -dit litgene /bin/bash"
3. Enter docker:
      d. "docker exec -it Litgene /bin/bash"
4. Inside docker go to the sample code solubility:
      e. "cd /home/tailab/LitGene/"
      f. open jupyter " jupyter notebook --ports 8888 --ip 0.0.0.0 --allow-root --no-browser"
5. In the system where docker is deployed:
      g. open browser "https://localhost:8888"
      h. enter the token 
      i. run the sample code "solubilityEval.ipynb"
6. or docker in remote system(optional):
       g.  open command on local system
       h. type "ssh -NL localhost:8888:localhost:8888 usernme@server"
       i. repeat step 5


 (If interested in Conda requirements provided in LitGene/dependencies tested on>

For other datafiles, models and output files . Please contact [asahu@salud.unm.edu]

