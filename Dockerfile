####### Dockerfile #######
FROM rocker/tidyverse:3.4.3

# Install OpenJDK-8
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk && \
    apt-get install -y ant && \
    apt-get clean;

# Fix certificate issues
RUN apt-get update && \
    apt-get install ca-certificates-java && \
    apt-get clean && \
    update-ca-certificates -f;

# Setup JAVA_HOME -- useful for docker commandline
ENV JAVA_HOME /usr/lib/jvm/java-8-openjdk-amd64/
RUN export JAVA_HOME

RUN apt-get update -qq && apt-get -y --no-install-recommends install \
	libglu1-mesa-dev \
&& install2.r --error \
    --deps TRUE \
    h2o 