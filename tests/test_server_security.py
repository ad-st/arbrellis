import http.client
import sys
import threading
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "server"))

from document_capture_server import make_handler  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402


class _Models:
    t5_enabled = False


class _Application:
    models = _Models()


class ServerSecurityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(_Application(), "secret"))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def request(self, path, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.server.server_port)
        connection.request("GET", path, headers=headers or {})
        response = connection.getresponse()
        body = response.read()
        connection.close()
        return response, body

    def test_api_rejects_missing_token(self):
        response, _ = self.request("/api/health")
        self.assertEqual(response.status, 401)

    def test_bearer_token_authorizes_health_check(self):
        response, body = self.request(
            "/api/health", {"Authorization": "Bearer secret"}
        )
        self.assertEqual(response.status, 200)
        self.assertIn(b'"t5gemma_enabled": false', body)
        self.assertEqual(response.getheader("X-Content-Type-Options"), "nosniff")

    def test_session_query_sets_protected_cookie(self):
        response, _ = self.request("/?session=secret")
        self.assertEqual(response.status, 200)
        cookie = response.getheader("Set-Cookie")
        self.assertIn("HttpOnly", cookie)
        self.assertIn("SameSite=Strict", cookie)


if __name__ == "__main__":
    unittest.main()
