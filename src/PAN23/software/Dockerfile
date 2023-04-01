from python:3.9.1
RUN mkdir -p /software/models
COPY *.py /software/
COPY ./models/*.pt /software/models
COPY requirements.txt /software/
RUN pip3 install -r /software/requirements.txt