import httpx


class PersonReIDClient:

    def __init__(self, host: str = "localhost", port: int = 4000):
        """
        Python Client for person re-identification API endpoints.

        Args:
            host (str, optional): API host address. Defaults to "localhost".
            port (int, optional): API port number. Defaults to 4000.
        """
        self.client = httpx.Client()
        self.endpoint = f"http://{host}:{port}"

    def ping(self) -> dict:
        """
        Check if API is up.

        Returns:
            dict: Returns {'message': 'pong'} if API is up.
        """
        response = self.client.get(f"{self.endpoint}/ping")
        return response.json()

    def infer(self, filepath: str) -> dict:
        """
        Run inference on image from image file path.

        Args:
            filepath (str): Image file path.

        Returns:
            dict: Predicted cluster.
        """
        files = {"files": open(filepath, "rb")}
        response = self.client.post(f"{self.endpoint}/infer", files = files)
        return response.json()
    
    def __del__(self):
        """
        Close TCP connections.
        """
        self.client.close()


if __name__ == '__main__':
    client = PersonReIDClient()
    print(client.ping())
    print(client.infer("Gallery/0_1_1000.jpg"))