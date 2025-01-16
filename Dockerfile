# python image
FROM python:3.11

#set the working directory in the container
WORKDIR /app

#copy the requirements.txt
COPY requirements.txt .

#intall dependencies
RUN pip install -r requirements.txt

#copy the application code
COPY . .

#expose the port for streamlit app
EXPOSE 8501

#run the streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false"]


