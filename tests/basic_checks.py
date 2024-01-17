import unittest
import os
import importlib.util

class TestLLMBench(unittest.TestCase):

    def test_library_ctranslate2_installed(self):
        """ Test if ctranslate2 library is installed """
        ctranslate2_installed = importlib.util.find_spec("ctranslate2") is not None
        self.assertTrue(ctranslate2_installed, "ctranslate2 library is not installed")

    def test_library_nvidia_installed(self):
        """ Test if nvidia library is installed """
        nvidia_installed = importlib.util.find_spec("nvidia") is not None
        self.assertTrue(nvidia_installed, "nvidia library is not installed")

    def test_library_torch_installed(self):
        """ Test if torch library is installed """
        torch_installed = importlib.util.find_spec("torch") is not None
        self.assertTrue(torch_installed, "torch library is not installed")

    def test_library_transformers_installed(self):
        """ Test if transformers library is installed """
        transformers_installed = importlib.util.find_spec("transformers") is not None
        self.assertTrue(transformers_installed, "transformers library is not installed")

    def test_library_vllm_installed(self):
        """ Test if vllm library is installed """
        vllm_installed = importlib.util.find_spec("vllm") is not None
        self.assertTrue(vllm_installed, "vllm library is not installed")

    def test_library_streamlit_installed(self):
        """ Test if streamlit library is installed """
        streamlit_installed = importlib.util.find_spec("streamlit") is not None
        self.assertTrue(streamlit_installed, "streamlit library is not installed")    
        
    def test_library_streamlitchat_installed(self):
        """ Test if streamlit-chat library is installed """
        streamlitchat_installed = importlib.util.find_spec("streamlit_chat") is not None
        self.assertTrue(streamlitchat_installed, "streamlit-chat library is not installed")    

    # def test_cuda_directory_exists(self):
    #     """ Test if CUDA directory exists """
    #     cuda_dir = '/'.join(importlib.util.find_spec("nvidia").origin.split('/')[:-1]) + '/cuda_runtime/lib/'
    #     self.assertTrue(os.path.isdir(cuda_dir), "CUDA directory does not exist")

    # def test_environment_variable_set(self):
    #     """ Test if LD_LIBRARY_PATH environment variable is set correctly """
    #     expected_path = '/'.join(importlib.util.find_spec("nvidia").origin.split('/')[:-1]) + '/cuda_runtime/lib/'
    #     ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
    #     self.assertEqual(ld_library_path, expected_path, "LD_LIBRARY_PATH environment variable is not set correctly")

if __name__ == '__main__':
    unittest.main()
