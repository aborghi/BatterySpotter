import argparse
from ibm_watson_machine_learning import APIClient

parser = argparse.ArgumentParser(prog="gui.py", description="Battery Spotter model deployment.")
parser.add_argument("--ibm_api_key", type=str, required=True)
parser.add_argument("--ibm_url", type=str, required=True)
parser.add_argument("--ibm_space_id", type=str, required=True)
parser.add_argument("--model_path", type=str, default="resources/efficient-d0-bat.tar.gz")
args = parser.parse_args()

wml_credentials = {
  "apikey": args.ibm_api_key,
  "url": args.ibm_url
}

client = APIClient(wml_credentials)

client.set.default_space(args.ibm_space_id)

software_spec_uid = client.software_specifications.get_id_by_name("runtime-22.2-py3.10")

metadata = {
    client.repository.ModelMetaNames.NAME: 'Battery Spotter TensorFlow model',
    client.repository.ModelMetaNames.TYPE: 'tensorflow_2.9',
    client.repository.ModelMetaNames.SOFTWARE_SPEC_UID: software_spec_uid
}

published_model = client.repository.store_model(model=args.model_path, meta_props=metadata)
published_model_uid = client.repository.get_model_id(published_model)
client.repository.list()

metadata = {
    client.deployments.ConfigurationMetaNames.NAME: "Deployment of Battery Spotter TensorFlow model",
    client.deployments.ConfigurationMetaNames.ONLINE: {}
}

created_deployment = client.deployments.create(published_model_uid, meta_props=metadata)
deployment_uid = client.deployments.get_id(created_deployment)

client.deployments.get_details(deployment_uid)
client.deployments.list()
