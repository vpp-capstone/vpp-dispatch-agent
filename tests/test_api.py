"""기능 테스트는 각 팀이 구현하면서 추가한다. 여기는 CI가 돌아가는지 확인하는 최소 테스트만 둔다."""


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}
