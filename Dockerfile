
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy the local files into the container
COPY . .

# Install any dependencies (make sure to have requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (optional, only if it’s a web app)
EXPOSE 8080

# Command to run your app
CMD ["python", "app.py"]
