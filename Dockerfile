FROM python:3.10

WORKDIR /src
# Copy the requirements file and install dependencies
COPY train_churn_model.py requirements.txt Customertravel.csv /src/
RUN pip install --no-cache-dir -r requirements.txt 

# Run the script
CMD ["python","train_churn_model.py"]