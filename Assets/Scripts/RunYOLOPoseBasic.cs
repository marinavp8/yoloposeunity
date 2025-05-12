using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using System.IO;
using System.Collections.Generic;

public class RunYOLOPoseBasic : MonoBehaviour
{
    [Header("Model Settings")]
    [Tooltip("Drag a YOLO11n-pose model .onnx file here")]
    public ModelAsset modelAsset;

    [Header("Display Settings")]
    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Header("Video Settings")]
    [Tooltip("Change this to the name of the video you put in the Assets/StreamingAssets folder")]
    public string videoFilename = "video.mp4";

    const BackendType backend = BackendType.GPUCompute;

    private Worker worker;
    private RenderTexture targetRT;
    private VideoPlayer video;

    private const int imageWidth = 640;
    private const int imageHeight = 640;

    List<Rect> detectedBoxes = new List<Rect>();
    List<List<Vector2>> detectedKeypoints = new List<List<Vector2>>();

    private Texture2D dotTexture;

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        LoadModel();
        SetupRenderTexture();
        SetupInput();
        CreateDotTexture();
    }

    void LoadModel()
    {
        var model = ModelLoader.Load(modelAsset);
        worker = new Worker(model, backend);

        Debug.Log("Model outputs:");
        foreach (var output in model.outputs)
        {
            Debug.Log($"Output name: {output.name}");
        }
    }

    void SetupRenderTexture()
    {
        targetRT = new RenderTexture(imageWidth, imageHeight, 0);
    }

    void SetupInput()
    {
        video = gameObject.AddComponent<VideoPlayer>();
        video.renderMode = VideoRenderMode.APIOnly;
        video.source = VideoSource.Url;
        video.url = Path.Join(Application.streamingAssetsPath, videoFilename);
        video.isLooping = true;
        video.Play();
    }

    private void Update()
    {
        ProcessFrame();

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    void ProcessFrame()
    {
        if (!video || !video.texture) return;

        RectTransform rt = displayImage.rectTransform;
        Vector2 rawImagePos = rt.position;
        Vector2 rawImageSize = rt.rect.size;

        float aspect = video.width * 1f / video.height;
        Graphics.Blit(video.texture, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
        displayImage.texture = targetRT;

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);
        worker.Schedule(inputTensor);

        using var output = (worker.PeekOutput("output0") as Tensor<float>).ReadbackAndClone();


        detectedBoxes.Clear();
        detectedKeypoints.Clear();
        List<Rect> allBoxes = new List<Rect>();
        List<List<Vector2>> allKeypoints = new List<List<Vector2>>();
        List<float> allScores = new List<float>();
        int numDetections = output.shape[2];
        int detectionSize = output.shape[1];
        float confidenceThreshold = 0.5f;
        for (int i = 0; i < numDetections; i++)
        {
            float confidence = output[0, 4, i];
            if (confidence > confidenceThreshold)
            {
                float x = output[0, 0, i];
                float y = output[0, 1, i];
                float w = output[0, 2, i];
                float h = output[0, 3, i];
                float px = rawImagePos.x - rawImageSize.x / 2 + (x / 640f) * rawImageSize.x;
                float py = rawImagePos.y - rawImageSize.y / 2 + (y / 640f) * rawImageSize.y;
                allBoxes.Add(new Rect(px - w / 2, py - h / 2, w, h));
                allScores.Add(confidence);
                List<Vector2> keypoints = new List<Vector2>();
                for (int k = 0; k < 17; k++)
                {
                    float kx = output[0, 5 + k * 3, i];
                    float ky = output[0, 5 + k * 3 + 1, i];
                    float kconf = output[0, 5 + k * 3 + 2, i];
                    if (kconf > 0.2f)
                    {
                        float kpx = rawImagePos.x - rawImageSize.x / 2 + (kx / 640f) * rawImageSize.x;
                        float kpy = rawImagePos.y - rawImageSize.y / 2 + (ky / 640f) * rawImageSize.y;
                        keypoints.Add(new Vector2(kpx, kpy));
                    }
                }
                allKeypoints.Add(keypoints);
            }
        }
        ApplyNMS(allBoxes, allScores, allKeypoints, 0.45f, out detectedBoxes, out detectedKeypoints);
        Debug.Log($"Personas detectadas en este frame: {detectedBoxes.Count}");
    }

    private void OnGUI()
    {
        foreach (var box in detectedBoxes)
            GUI.Box(new Rect(box.x, box.y, box.width, box.height), "");

        foreach (var keypoints in detectedKeypoints)
            foreach (var kp in keypoints)
                GUI.DrawTexture(new Rect(kp.x - 2, kp.y - 2, 4, 4), dotTexture);
    }

    private void OnDestroy()
    {
        worker?.Dispose();
    }

    void CreateDotTexture()
    {
        dotTexture = new Texture2D(4, 4);
        Color32[] pixels = new Color32[16];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = Color.white;
        dotTexture.SetPixels32(pixels);
        dotTexture.Apply();
    }

    float IoU(Rect a, Rect b)
    {
        float x1 = Mathf.Max(a.xMin, b.xMin);
        float y1 = Mathf.Max(a.yMin, b.yMin);
        float x2 = Mathf.Min(a.xMax, b.xMax);
        float y2 = Mathf.Min(a.yMax, b.yMax);
        float interArea = Mathf.Max(0, x2 - x1) * Mathf.Max(0, y2 - y1);
        float unionArea = a.width * a.height + b.width * b.height - interArea;
        return interArea / (unionArea + 1e-6f);
    }

    void ApplyNMS(List<Rect> boxes, List<float> scores, List<List<Vector2>> keypoints, float iouThreshold, out List<Rect> nmsBoxes, out List<List<Vector2>> nmsKeypoints)
    {
        nmsBoxes = new List<Rect>();
        nmsKeypoints = new List<List<Vector2>>();
        List<int> idxs = new List<int>();
        for (int i = 0; i < boxes.Count; i++) idxs.Add(i);
        idxs.Sort((i, j) => scores[j].CompareTo(scores[i]));
        bool[] removed = new bool[boxes.Count];
        for (int i = 0; i < idxs.Count; i++)
        {
            int idx = idxs[i];
            if (removed[idx]) continue;
            nmsBoxes.Add(boxes[idx]);
            nmsKeypoints.Add(keypoints[idx]);
            for (int j = i + 1; j < idxs.Count; j++)
            {
                int idx2 = idxs[j];
                if (removed[idx2]) continue;
                if (IoU(boxes[idx], boxes[idx2]) > iouThreshold)
                    removed[idx2] = true;
            }
        }
    }
}