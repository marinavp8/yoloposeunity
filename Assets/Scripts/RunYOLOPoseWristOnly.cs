using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using System.IO;
using System.Collections.Generic;

public class RunYOLOPoseWristOnly : MonoBehaviour
{
    [Header("Model Settings")]
    public ModelAsset poseModelAsset;
    public ModelAsset handModelAsset;

    [Header("Display Settings")]
    public RawImage displayImage;
    public Sprite boxSprite;
    public Sprite[] markerTextures;

    [Header("Video Settings")]
    public string videoFilename = "video.mp4";

    public enum InputType { Video, Webcam }
    [Header("Input Type")]
    public InputType inputType = InputType.Video;

    const BackendType backend = BackendType.GPUCompute;

    private Worker poseWorker;
    private Worker handWorker;
    private RenderTexture targetRT;
    private VideoPlayer video;
    private WebCamTexture webcam;

    private const int imageWidth = 640;
    private const int imageHeight = 640;
    private const int handSize = 224;

    private Texture2D boxTexture;
    private List<Vector2> detectedWrists = new List<Vector2>();
    private const int boxSize = 300;

    private List<List<GameObject>> landmarkPools = new List<List<GameObject>>();

    private struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
    }

    private struct WristKeypoint
    {
        public Vector2 position;
        public float confidence;
        public WristKeypoint(Vector2 pos, float conf)
        {
            position = pos;
            confidence = conf;
        }
    }

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;
        LoadModels();
        SetupRenderTexture();
        SetupInput();
        CreateBoxTexture();
    }

    void LoadModels()
    {
        var poseModel = ModelLoader.Load(poseModelAsset);
        poseWorker = new Worker(poseModel, backend);

        var handModel = ModelLoader.Load(handModelAsset);
        handWorker = new Worker(handModel, backend);
    }

    void SetupRenderTexture()
    {
        targetRT = new RenderTexture(imageWidth, imageHeight, 0);
    }

    void SetupInput()
    {
        if (inputType == InputType.Webcam)
        {
            webcam = new WebCamTexture();
            webcam.Play();
        }
        else
        {
            video = gameObject.AddComponent<VideoPlayer>();
            video.renderMode = VideoRenderMode.APIOnly;
            video.source = VideoSource.Url;
            video.url = Path.Join(Application.streamingAssetsPath, videoFilename);
            video.isLooping = true;
            video.Play();
        }
    }

    void Update()
    {
        ProcessFrame();

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    void ProcessFrame()
    {
        Texture sourceTexture = null;
        if (inputType == InputType.Webcam)
        {
            if (webcam == null || !webcam.isPlaying || !webcam.didUpdateThisFrame) return;
            sourceTexture = webcam;
        }
        else
        {
            if (!video || !video.texture) return;
            sourceTexture = video.texture;
        }

        RectTransform rt = displayImage.rectTransform;
        Vector2 rawImagePos = rt.position;
        Vector2 rawImageSize = rt.rect.size;

        float aspect = (float)sourceTexture.width / sourceTexture.height;
        Graphics.Blit(sourceTexture, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
        displayImage.texture = targetRT;

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);
        poseWorker.Schedule(inputTensor);

        using var output = (poseWorker.PeekOutput("output0") as Tensor<float>).ReadbackAndClone();

        List<WristKeypoint> allWrists = new List<WristKeypoint>();

        int numDetections = output.shape[2];
        float confidenceThreshold = 0.7f;

        for (int i = 0; i < numDetections; i++)
        {
            float confidence = output[0, 4, i];
            if (confidence < confidenceThreshold)
                continue;

            float cx = output[0, 0, i];
            float cy = output[0, 1, i];
            float w = output[0, 2, i];
            float h = output[0, 3, i];

            Vector2 bboxCenter = ToScreenCoords(cx, cy, rawImagePos, rawImageSize);

            int leftWristIndex = 9;
            int rightWristIndex = 10;

            float lx = output[0, 5 + leftWristIndex * 3, i];
            float ly = output[0, 5 + leftWristIndex * 3 + 1, i];
            float lconf = output[0, 5 + leftWristIndex * 3 + 2, i];

            float rx = output[0, 5 + rightWristIndex * 3, i];
            float ry = output[0, 5 + rightWristIndex * 3 + 1, i];
            float rconf = output[0, 5 + rightWristIndex * 3 + 2, i];

            if (lconf > 0.2f)
            {
                Vector2 pos = ToScreenCoords(lx, ly, rawImagePos, rawImageSize);
                allWrists.Add(new WristKeypoint(pos, lconf));
            }

            if (rconf > 0.2f)
            {
                Vector2 pos = ToScreenCoords(rx, ry, rawImagePos, rawImageSize);
                allWrists.Add(new WristKeypoint(pos, rconf));
            }
        }

        detectedWrists.Clear();
        foreach (var kp in ApplyNMS(allWrists, 60f))
            detectedWrists.Add(kp.position);

        ClearLandmarks();
        for (int i = 0; i < detectedWrists.Count; i++)
        {
            Vector2 wrist = detectedWrists[i];
            ProcessHandLandmarks(wrist, rawImagePos, rawImageSize, i);
        }
    }

    void ProcessHandLandmarks(Vector2 wristPos, Vector2 rawImagePos, Vector2 rawImageSize, int handIndex)
    {
        float centerX = Mathf.Clamp(wristPos.x, boxSize / 2, imageWidth - boxSize / 2);
        float centerY = Mathf.Clamp(wristPos.y, boxSize / 2, imageHeight - boxSize / 2);

        float x = centerX - boxSize / 2;
        float y = centerY - boxSize / 2;

        RenderTexture handRT = new RenderTexture(handSize, handSize, 0);
        Texture2D tempTex = new Texture2D(boxSize, boxSize);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = targetRT;
        tempTex.ReadPixels(new Rect(x, y, boxSize, boxSize), 0, 0);
        tempTex.Apply();
        RenderTexture.active = prev;

        Graphics.Blit(tempTex, handRT);

        using var inputTensor = new Tensor<float>(new TensorShape(1, 3, handSize, handSize));
        TextureConverter.ToTensor(handRT, inputTensor, default);
        handWorker.Schedule(inputTensor);

        using var landmarks = (handWorker.PeekOutput("Identity") as Tensor<float>).ReadbackAndClone();

        float scale = (float)boxSize / handSize;

        for (int j = 0; j < 21; j++)
        {
            float landmarkX = landmarks[0, j * 3];
            float landmarkY = landmarks[0, j * 3 + 1];

            float transformedX = x + (landmarkX * scale);
            float transformedY = y + (landmarkY * scale);

            var marker = new BoundingBox
            {
                centerX = transformedX,
                centerY = transformedY,
                width = 8f,
                height = 8f,
            };
            DrawLandmark(marker, j < markerTextures.Length ? markerTextures[j] : boxSprite, handIndex, j);
        }

        Destroy(tempTex);
        RenderTexture.active = null;
        handRT.Release();
    }

    void DrawHandLandmarks(Tensor<float> landmarks, float x, float y, float width, float height, int handIndex)
    {
        float scale = (float)width / handSize;
        while (landmarkPools.Count <= handIndex)
            landmarkPools.Add(new List<GameObject>());
        var pool = landmarkPools[handIndex];
        for (int j = 0; j < 21; j++)
        {
            var marker = new BoundingBox
            {
                centerX = x + (landmarks[0, j * 3] * scale),
                centerY = y + (landmarks[0, j * 3 + 1] * scale),
                width = 8f,
                height = 8f,
            };
            DrawLandmark(marker, j < markerTextures.Length ? markerTextures[j] : boxSprite, handIndex, j);
        }
    }

    void DrawLandmark(BoundingBox box, Sprite sprite, int handIndex, int landmarkIndex)
    {
        while (landmarkPools.Count <= handIndex)
            landmarkPools.Add(new List<GameObject>());
        var pool = landmarkPools[handIndex];
        while (pool.Count <= landmarkIndex)
        {
            var panel = new GameObject($"landmark_{handIndex}_{pool.Count}");
            panel.AddComponent<CanvasRenderer>();
            panel.AddComponent<Image>();
            panel.transform.SetParent(displayImage.transform, false);
            pool.Add(panel);
        }
        GameObject panelObj = pool[landmarkIndex];
        panelObj.SetActive(true);
        var img = panelObj.GetComponent<Image>();
        img.color = Color.white;
        img.sprite = sprite;
        img.type = Image.Type.Sliced;
        Vector2 localPos = PixelToRawImageLocal(box.centerX, box.centerY);
        panelObj.transform.localPosition = new Vector3(localPos.x, localPos.y, 0);
        RectTransform rt = panelObj.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width, box.height);
    }

    void ClearLandmarks()
    {
        foreach (var pool in landmarkPools)
            foreach (var obj in pool)
                obj.SetActive(false);
    }

    Vector2 ToScreenCoords(float kx, float ky, Vector2 rawImagePos, Vector2 rawImageSize)
    {
        float px = rawImagePos.x - rawImageSize.x / 2 + (kx / 640f) * rawImageSize.x;
        float py = rawImagePos.y - rawImageSize.y / 2 + (ky / 640f) * rawImageSize.y;
        return new Vector2(px, py);
    }

    void OnGUI()
    {
        foreach (var kp in detectedWrists)
        {
            float x = kp.x - boxSize / 2;
            float y = kp.y - boxSize;
            GUI.DrawTexture(new Rect(x, y, boxSize, boxSize), boxTexture);
        }
    }

    void OnDestroy()
    {
        poseWorker?.Dispose();
        handWorker?.Dispose();
        if (webcam != null)
        {
            webcam.Stop();
        }
        ClearLandmarks();
    }

    void CreateBoxTexture()
    {
        boxTexture = new Texture2D(boxSize, boxSize);
        Color32[] pixels = new Color32[boxSize * boxSize];
        for (int i = 0; i < pixels.Length; i++)
        {
            int x = i % boxSize;
            int y = i / boxSize;
            if (x < 2 || x > boxSize - 3 || y < 2 || y > boxSize - 3)
            {
                pixels[i] = Color.green;
            }
            else
            {
                pixels[i] = new Color32(0, 0, 0, 0);
            }
        }
        boxTexture.SetPixels32(pixels);
        boxTexture.Apply();
    }

    List<WristKeypoint> ApplyNMS(List<WristKeypoint> wrists, float distThreshold)
    {
        List<WristKeypoint> result = new List<WristKeypoint>();
        wrists.Sort((a, b) => b.confidence.CompareTo(a.confidence));
        bool[] suppressed = new bool[wrists.Count];

        for (int i = 0; i < wrists.Count; i++)
        {
            if (suppressed[i]) continue;
            result.Add(wrists[i]);
            for (int j = i + 1; j < wrists.Count; j++)
            {
                if (!suppressed[j] && Vector2.Distance(wrists[i].position, wrists[j].position) < distThreshold)
                {
                    suppressed[j] = true;
                }
            }
        }
        return result;
    }

    Vector2 PixelToRawImageLocal(float px, float py)
    {
        float localX = px - (imageWidth / 2f);
        float localY = (imageHeight / 2f) - py;
        return new Vector2(localX, localY);
    }
}
