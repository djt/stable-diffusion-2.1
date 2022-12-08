from sanic import Sanic, response
import subprocess
import app as user_src

user_src.init()

server = Sanic("my_app")

@server.route('/healthcheck', methods=["GET"])
def healthcheck(request):
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0: # success state on shell command
        gpu = True

    return response.json({"state": "healthy", "gpu": gpu})

@server.route('/', methods=["POST"]) 
def inference(request):
    try:
        model_inputs = response.json.loads(request.json)
    except:
        model_inputs = request.json

    output = user_src.inference(model_inputs)

    return response.json(output)


if __name__ == '__main__':
    server.run(host='0.0.0.0', port="8000", workers=1)
