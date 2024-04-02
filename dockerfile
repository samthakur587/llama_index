# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY gcpreq.txt .
COPY otherreq.txt .



# Install the dependencies
RUN pip install  -r gcpreq.txt
RUN pip install -r otherreq.txt

# Copy the rest of the application code into the container
COPY . .
EXPOSE 8000
# Set the command to run the application
CMD ["python","aap.py"]