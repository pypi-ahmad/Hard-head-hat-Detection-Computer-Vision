import types
import pytest


class FakeSidebar:
    def __init__(self):
        self.errors = []

    def error(self, message):
        self.errors.append(message)


@pytest.fixture
def fake_streamlit():
    st = types.SimpleNamespace()
    st.sidebar = FakeSidebar()
    st.set_page_config = lambda **kwargs: None
    st.warning = lambda *args, **kwargs: None
    st.success = lambda *args, **kwargs: None
    st.title = lambda *args, **kwargs: None
    st.radio = lambda *args, **kwargs: "Image"
    st.file_uploader = lambda *args, **kwargs: None
    st.button = lambda *args, **kwargs: False
    st.image = lambda *args, **kwargs: None
    st.columns = lambda *args, **kwargs: [types.SimpleNamespace(metric=lambda *a, **k: None,
                                                                 error=lambda *a, **k: None,
                                                                 success=lambda *a, **k: None),
                                           types.SimpleNamespace(metric=lambda *a, **k: None,
                                                                 error=lambda *a, **k: None,
                                                                 success=lambda *a, **k: None)]
    st.empty = lambda: types.SimpleNamespace(image=lambda *a, **k: None,
                                             container=lambda: types.SimpleNamespace(__enter__=lambda s: s,
                                                                                     __exit__=lambda *e: False))
    st.checkbox = lambda *args, **kwargs: False
    st.select_slider = lambda *args, **kwargs: 1.0
    st.metric = lambda *args, **kwargs: None
    return st
