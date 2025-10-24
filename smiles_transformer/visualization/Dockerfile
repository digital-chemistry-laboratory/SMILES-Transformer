FROM python:3.10.4

COPY ./plotly_script.py ./plotly_script.py

RUN pip install CGRtools
RUN pip install dash
RUN pip install pandas
RUN pip install py-mini-racer

EXPOSE 8050

CMD ["python", "./plotly_script.py"]
