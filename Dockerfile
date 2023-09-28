ARG BASE_IMAGE
FROM $BASE_IMAGE as base

ENV SHELL /bin/bash
SHELL [ "/bin/bash", "-c" ]

ENV DEBIAN_FRONTEND=noninteractive

RUN echo ${DEBIAN_FRONTEND}

RUN apt-get update &&\ 
    apt-get install --no-install-recommends -y \
    python-opengl

RUN pip3 install \
    gym==0.9.6 \
    tensorflow \
    Pillow \
    matplotlib \
    jupyter \
    pytest \
    docopt \
    pyyaml \
    protobuf \
    grpcio \
    pandas \
    scipy \
    ipykernel \
    pyglet==1.5.27 \
    box2d

COPY python /root/python
RUN pip3 install /root/python