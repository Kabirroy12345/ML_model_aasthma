FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port the app runs on (optional, if using Flask)
EXPOSE 5000

# Command to run the application
CMD ["python", "app.py"]
