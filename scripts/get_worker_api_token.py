import hmac
import hashlib

def get_worker_api_token(
    message: str,
    secret: str,
):
    scope = 'generate_access_token'

    if isinstance(secret, str):
        secret = secret.encode()

    message = repr((scope, message))
    return hmac.new(
        secret,
        message.encode(),
        hashlib.sha256,
    ).hexdigest()

if __name__ == '__main__':
    worker_name: str = 'Kaggle Worker `crows.nony1@gmail.com`'
    worker_code: str = 'none'
    secret: str = 'd7e28c67-de44-41e7-830d-dc0f8129d5cd'
    values = (worker_name, worker_code,)
    token_str = '|'.join(
        str(val) for val in values
    )
    print(get_worker_api_token(
        token_str,
        secret,
    ))