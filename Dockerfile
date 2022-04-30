FROM ubuntu:20.04
WORKDIR /
COPY client.py ./
COPY requirements.txt /tmp/
RUN apt-get update && apt-get install -y python3 python3-pip 
RUN pip3 install -r /tmp/requirements.txt
EXPOSE 8000
CMD ["python3", "client.py"]

