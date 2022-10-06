# FLKube

## Steps:

##How to replicate our paper experiments?

* conda activate <fedlab-environment> according to requirements.txt
* `./launch_eg_rodrigo.sh` notice that inside this script there are many parameters;
* as long as the experiments run, you must change the CNN inside client and server

* Build Docker for Clientes
`docker build -t romoreira/flkube-client`

* Run the Container with Built image
`docker run -p80:8080 romoreira/flkube-client`
