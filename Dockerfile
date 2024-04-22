FROM python:3.10.14-bookworm

COPY . GradioApp

WORKDIR GradioApp

RUN pip install pip -U
RUN pip install -r Requirements.txt

EXPOSE 8081

CMD [ "python", "main.py" ]