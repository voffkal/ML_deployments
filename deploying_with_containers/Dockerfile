FROM python:3.11

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /opt/houseprice_api

ADD ./houseprice_api /opt/houseprice_api

# The model package is downloaded from its GitHub Release before the build and
# installed from the local wheel, so the image records exactly which trained
# model it ships.
COPY model_pkg/*.whl /tmp/

RUN pip install --upgrade pip
RUN pip install /tmp/*.whl
RUN pip install -r /opt/houseprice_api/requirements.txt

RUN chmod +x /opt/houseprice_api/run.sh
RUN chown -R ml-api-user:ml-api-user ./

USER ml-api-user

ENV PORT=8001

EXPOSE 8001

CMD ["bash", "./run.sh"]