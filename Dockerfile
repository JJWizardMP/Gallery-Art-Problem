FROM jupyter/base-notebook:python-3.10

# Copy the requirements.txt file
COPY requirements.txt /tmp/

# Install dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# RUN pip install numpy pandas scipy matplotlib shapely

# Copy your project files into the Docker image
COPY . /usr/src/app

# Set the working directory to where your project is copied
WORKDIR /usr/src/app