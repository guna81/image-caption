# Specify the base image
FROM python:3.10

# Set the working directory
WORKDIR /image-caption

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
# RUN pip install --no-cache-dir -r requirements.txt
RUN pip install -r requirements.txt

# Copy the application code
COPY . .

# Expose the port
EXPOSE 5002

# Define the command to run the application
CMD ["python", "app.py"]
