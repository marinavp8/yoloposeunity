using UnityEngine;
using Unity.Sentis;
using UnityEngine.Video;
using UnityEngine.UI;
using System.IO;
using System.Collections.Generic;

/*
 *                   Hand Landmarks Inference
 *                   ========================
 *                   
 * Basic inference script for blaze hand landmarks
 * 
 * Put this script on the Main Camera
 * Drag the sentis file onto the modelAsset field
 * Create a RawImage of in the scene
 * Put a link to that image in previewUI
 * Put a video in Assets/StreamingAssets folder and put the name of it int videoName
 * Or put a test image in inputImage
 * Set inputType to appropriate input
 */

public class RunHandLandmark : MonoBehaviour
{
    //Draw the *.sentis or *.onnx model asset here:
    public ModelAsset asset;
    string modelName = "hand_landmark.sentis";
    //Drag a link to a raw image here:
    public RawImage previewUI = null;

    // Put your bounding box sprite image here
    public Sprite boxSprite;

    // 6 optional sprite images (left eye, right eye, nose, mouth, left ear, right ear)
    public Sprite[] markerTextures;

    public string videoName = "chatting.mp4";

    public Texture2D inputImage;

    public InputType inputType = InputType.Video;

    //Resolution of preview image or video
    Vector2Int resolution = new Vector2Int(640, 640);
    WebCamTexture webcam;
    VideoPlayer video;

    const BackendType backend = BackendType.GPUCompute;

    RenderTexture targetTexture;
    public enum InputType { Image, Video, Webcam };

    Worker worker;

    //Holds image size
    const int size = 224;

    Model model;

    //webcam device name:
    const string deviceName = "";

    bool closing = false;

    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
    }

    List<GameObject> boxPool = new();
    void Start()
    {
        //(Note: if using a webcam on mobile get permissions here first)

        targetTexture = new RenderTexture(resolution.x, resolution.y, 0);
        previewUI.texture = targetTexture;

        SetupInput();
        SetupModel();
        SetupEngine();
    }

    void SetupModel()
    {
        model = ModelLoader.Load(asset);
        //model = ModelLoader.Load(Path.Join(Application.streamingAssetsPath ,modelName));
    }
    public void SetupEngine()
    {
        worker = new Worker(model, backend);
    }

    void SetupInput()
    {
        switch (inputType)
        {
            case InputType.Webcam:
                {
                    webcam = new WebCamTexture(deviceName, resolution.x, resolution.y);
                    webcam.requestedFPS = 30;
                    webcam.Play();
                    break;
                }
            case InputType.Video:
                {
                    video = gameObject.AddComponent<VideoPlayer>();
                    video.renderMode = VideoRenderMode.APIOnly;
                    video.source = VideoSource.Url;
                    video.url = Path.Combine(Application.streamingAssetsPath, videoName);
                    video.isLooping = true;
                    video.Play();
                    break;
                }
            default:
                {
                    Graphics.Blit(inputImage, targetTexture);
                }
                break;
        }
    }

    void Update()
    {
        if (inputType == InputType.Webcam)
        {
            // Format video input
            if (!webcam.didUpdateThisFrame) return;

            var aspect1 = (float)webcam.width / webcam.height;
            var aspect2 = (float)resolution.x / resolution.y;
            var gap = aspect2 / aspect1;

            var vflip = webcam.videoVerticallyMirrored;
            var scale = new Vector2(gap, vflip ? -1 : 1);
            var offset = new Vector2((1 - gap) / 2, vflip ? 1 : 0);

            Graphics.Blit(webcam, targetTexture, scale, offset);
        }
        if (inputType == InputType.Video)
        {
            var aspect1 = (float)video.width / video.height;
            var aspect2 = (float)resolution.x / resolution.y;
            var gap = aspect2 / aspect1;

            var vflip = false;
            var scale = new Vector2(gap, vflip ? -1 : 1);
            var offset = new Vector2((1 - gap) / 2, vflip ? 1 : 0);
            Graphics.Blit(video.texture, targetTexture, scale, offset);
        }
        if (inputType == InputType.Image)
        {
            Graphics.Blit(inputImage, targetTexture);
        }

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            closing = true;
            Application.Quit();
        }

        if (Input.GetKeyDown(KeyCode.P))
        {
            previewUI.enabled = !previewUI.enabled;
        }
    }

    void LateUpdate()
    {
        if (!closing)
        {
            RunInference(targetTexture);
        }
    }

    void DrawLandmarks(Tensor<float> landmarks, Vector2 scale)
    {
        //Draw the landmarks on the hand
        for (int j = 0; j < 21; j++)
        {
            var marker = new BoundingBox
            {
                centerX = landmarks[0, j * 3] * scale.x - (size / 2) * scale.x,
                centerY = landmarks[0, j * 3 + 1] * scale.y - (size / 2) * scale.y,
                width = 8f * scale.x,
                height = 8f * scale.y,
            };
            DrawBox(marker, j < markerTextures.Length ? markerTextures[j] : boxSprite, j);
        }
    }

    void RunInference(Texture source)
    {
        using var inputTensor = new Tensor<float>(new TensorShape(1, 3, size, size));
        TextureConverter.ToTensor(source, inputTensor, default);

        worker.Schedule(inputTensor);

        using var landmarks = (worker.PeekOutput("Identity") as Tensor<float>).ReadbackAndClone();

        ClearAnnotations();

        Vector2 markerScale = previewUI.rectTransform.rect.size / size;
        DrawLandmarks(landmarks, markerScale);

        bool showExtraInformation = false;
        if (showExtraInformation)
        {
            using var A = (worker.PeekOutput("Identity_1") as Tensor<float>).ReadbackAndClone();
            using var B = (worker.PeekOutput("Identity_2") as Tensor<float>).ReadbackAndClone();
            Debug.Log("A,B=" + A[0, 0] + "," + B[0, 0]);
        }
    }

    public void DrawBox(BoundingBox box, Sprite sprite, int ID)
    {
        GameObject panel = null;
        if (ID >= boxPool.Count)
        {
            panel = new GameObject("landmark");
            panel.AddComponent<CanvasRenderer>();
            panel.AddComponent<Image>();
            panel.transform.SetParent(previewUI.transform, false);
            boxPool.Add(panel);
        }
        else
        {
            panel = boxPool[ID];
            panel.SetActive(true);
        }

        var img = panel.GetComponent<Image>();
        img.color = Color.white;
        img.sprite = sprite;
        img.type = Image.Type.Sliced;

        panel.transform.localPosition = new Vector3(box.centerX, -box.centerY);
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);
    }

    public void ClearAnnotations()
    {
        for (int i = 0; i < boxPool.Count; i++)
        {
            boxPool[i].SetActive(false);
        }
    }

    void CleanUp()
    {
        closing = true;
        if (webcam) Destroy(webcam);
        if (video) Destroy(video);
        RenderTexture.active = null;
        targetTexture.Release();
        worker?.Dispose();
        worker = null;
    }

    void OnDestroy()
    {
        CleanUp();
    }
}

