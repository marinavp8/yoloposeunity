using System.Collections.Generic;
using Unity.Sentis;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Video;
using System.IO;
using FF = Unity.Sentis.Functional;
using System.Linq;

public class RunYOLOPose : MonoBehaviour
{
    [Tooltip("Drag a YOLOv8n-pose model .onnx file here")]
    public ModelAsset modelAsset;

    [Tooltip("Create a Raw Image in the scene and link it here")]
    public RawImage displayImage;

    [Tooltip("Drag a border box texture here")]
    public Texture2D borderTexture;

    [Tooltip("Select an appropriate font for the labels")]
    public Font font;

    [Tooltip("Change this to the name of the video you put in the Assets/StreamingAssets folder")]
    public string videoFilename = "video.mp4";

    [Tooltip("Color for keypoint visualization")]
    public Color keypointColor = Color.red;

    [Tooltip("Size of keypoint visualization")]
    public float keypointSize = 5f;

    [Tooltip("Color for skeleton lines")]
    public Color skeletonColor = Color.green;

    [Tooltip("Width of skeleton lines")]
    public float skeletonWidth = 2f;

    const BackendType backend = BackendType.GPUCompute;

    private Transform displayLocation;
    private Worker worker;
    private RenderTexture targetRT;
    private Sprite borderSprite;

    //Image size for the model
    private const int imageWidth = 640;
    private const int imageHeight = 640;

    private VideoPlayer video;

    List<GameObject> boxPool = new();
    List<GameObject> keypointPool = new();
    List<GameObject> skeletonPool = new();

    [Tooltip("Intersection over union threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)] float iouThreshold = 0.2f;

    [Tooltip("Confidence score threshold used for non-maximum suppression")]
    [SerializeField, Range(0, 1)] float scoreThreshold = 0.2f;

    Tensor<float> centersToCorners;

    private readonly string[] keypointNames = new string[]
    {
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"
    };

    private readonly (int, int)[] skeletonConnections = new (int, int)[]
    {
        (5, 7), (7, 9),   // Left arm
        (6, 8), (8, 10),  // Right arm
        (5, 6),           // Shoulders
        (5, 11), (6, 12), // Torso
        (11, 13), (13, 15), // Left leg
        (12, 14), (14, 16)  // Right leg
    };

    public struct BoundingBox
    {
        public float centerX;
        public float centerY;
        public float width;
        public float height;
        public float[] keypoints; // Array of 51 values (17 keypoints * 3 values each [x,y,confidence])
    }

    void Start()
    {
        Application.targetFrameRate = 60;
        Screen.orientation = ScreenOrientation.LandscapeLeft;

        LoadModel();

        targetRT = new RenderTexture(imageWidth, imageHeight, 0);

        displayLocation = displayImage.transform;

        SetupInput();

        borderSprite = Sprite.Create(borderTexture, new Rect(0, 0, borderTexture.width, borderTexture.height), new Vector2(borderTexture.width / 2, borderTexture.height / 2));
    }

    void LoadModel()
    {
        //Load model
        var model1 = ModelLoader.Load(modelAsset);

        centersToCorners = new Tensor<float>(new TensorShape(4, 4),
        new float[]
        {
                    1,      0,      1,      0,
                    0,      1,      0,      1,
                    -0.5f,  0,      0.5f,   0,
                    0,      -0.5f,  0,      0.5f
        });

        // Procesar directamente la salida del modelo, sin NMS
        var graph = new FunctionalGraph();
        var inputs = graph.AddInputs(model1);
        var modelOutput = FF.Forward(model1, inputs)[0];                        //shape=(1,56,8400) para pose model
        var boxCoords = modelOutput[0, 0..4, ..].Transpose(0, 1);               //shape=(8400,4)
        var keypoints = modelOutput[0, 4.., ..].Transpose(0, 1);                //shape=(8400,51)

        // Crear worker para correr el modelo (sin NMS)
        worker = new Worker(graph.Compile(boxCoords, keypoints), backend);
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
        ExecuteML();

        if (Input.GetKeyDown(KeyCode.Escape))
        {
            Application.Quit();
        }
    }

    public void ExecuteML()
    {
        ClearAnnotations();

        if (video && video.texture)
        {
            float aspect = video.width * 1f / video.height;
            Graphics.Blit(video.texture, targetRT, new Vector2(1f / aspect, 1), new Vector2(0, 0));
            displayImage.texture = targetRT;
        }
        else return;

        using Tensor<float> inputTensor = new Tensor<float>(new TensorShape(1, 3, imageHeight, imageWidth));
        TextureConverter.ToTensor(targetRT, inputTensor, default);
        worker.Schedule(inputTensor);

        using var output = (worker.PeekOutput("output_0") as Tensor<float>).ReadbackAndClone();
        Debug.Log($"output.shape: {string.Join(", ", output.shape)}");
        using var keypoints = (worker.PeekOutput("output_1") as Tensor<float>).ReadbackAndClone();
        Debug.Log($"keypoints.shape: {string.Join(", ", keypoints.shape)}");

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        float scaleX = displayWidth / imageWidth;
        float scaleY = displayHeight / imageHeight;
        int boxesFound = output.shape[0];
        Debug.Log($"Found {boxesFound} persons in the frame");

        for (int n = 0; n < Mathf.Min(boxesFound, 200); n++)
        {
            Debug.Log($"Box raw: cx={output[n, 0]}, cy={output[n, 1]}, w={output[n, 2]}, h={output[n, 3]}");

            var box = new BoundingBox
            {
                centerX = output[n, 0],
                centerY = output[n, 1],
                width = output[n, 2],
                height = output[n, 3],
                keypoints = new float[51]
            };

            for (int k = 0; k < 51; k++)
            {
                box.keypoints[k] = keypoints[n, k];
            }

            string kpStr = string.Join(", ", Enumerable.Range(0, 52).Select(i => keypoints[0, i].ToString("F2")));
            Debug.Log($"Keypoints raw: {kpStr}");

            Debug.Log($"Person {n + 1}: at position ({box.centerX:F2}, {box.centerY:F2}) with size {box.width:F2}x{box.height:F2}");
            DrawBox(box, n, displayHeight * 0.05f);
            DrawKeypoints(box, n);
            DrawSkeleton(box, n);
        }
    }

    public void DrawBox(BoundingBox box, int id, float fontSize)
    {
        //Create the bounding box graphic or get from pool
        GameObject panel;
        if (id < boxPool.Count)
        {
            panel = boxPool[id];
            panel.SetActive(true);
        }
        else
        {
            panel = CreateNewBox(Color.yellow);
        }

        // Scale coordinates to display dimensions
        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        // Scale the coordinates to match the video aspect ratio
        float videoAspect = (float)video.width / video.height;
        float displayAspect = displayWidth / displayHeight;
        float scaleX = displayWidth;
        float scaleY = displayHeight;

        if (videoAspect > displayAspect)
        {
            scaleY = displayWidth / videoAspect;
        }
        else
        {
            scaleX = displayHeight * videoAspect;
        }

        float scaledX = box.centerX * scaleX - scaleX / 2;
        float scaledY = -box.centerY * scaleY + scaleY / 2;

        //Set box position
        panel.transform.localPosition = new Vector3(scaledX, scaledY, 0);

        //Set box size
        RectTransform rt = panel.GetComponent<RectTransform>();
        rt.sizeDelta = new Vector2(box.width * scaleX, box.height * scaleY);

        //Set label text
        var label = panel.GetComponentInChildren<Text>();
        label.text = "Person";
        label.fontSize = (int)fontSize;
    }

    public GameObject CreateNewBox(Color color)
    {
        var panel = new GameObject("ObjectBox");
        panel.AddComponent<CanvasRenderer>();
        Image img = panel.AddComponent<Image>();
        img.color = color;
        img.sprite = borderSprite;
        img.type = Image.Type.Sliced;
        panel.transform.SetParent(displayLocation, false);

        var text = new GameObject("ObjectLabel");
        text.AddComponent<CanvasRenderer>();
        text.transform.SetParent(panel.transform, false);
        Text txt = text.AddComponent<Text>();
        txt.font = font;
        txt.color = color;
        txt.fontSize = 40;
        txt.horizontalOverflow = HorizontalWrapMode.Overflow;

        RectTransform rt2 = text.GetComponent<RectTransform>();
        rt2.offsetMin = new Vector2(20, rt2.offsetMin.y);
        rt2.offsetMax = new Vector2(0, rt2.offsetMax.y);
        rt2.offsetMin = new Vector2(rt2.offsetMin.x, 0);
        rt2.offsetMax = new Vector2(rt2.offsetMax.x, 30);
        rt2.anchorMin = new Vector2(0, 0);
        rt2.anchorMax = new Vector2(1, 1);

        boxPool.Add(panel);
        return panel;
    }

    public void DrawKeypoints(BoundingBox box, int id)
    {
        // Create or get keypoint objects from pool
        int requiredKeypoints = (id + 1) * 17;
        while (keypointPool.Count < requiredKeypoints)
        {
            var keypoint = new GameObject("Keypoint");
            keypoint.AddComponent<CanvasRenderer>();
            Image img = keypoint.AddComponent<Image>();
            img.color = keypointColor;
            keypoint.transform.SetParent(displayLocation, false);
            keypointPool.Add(keypoint);
        }

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        // Scale the coordinates to match the video aspect ratio
        float videoAspect = (float)video.width / video.height;
        float displayAspect = displayWidth / displayHeight;
        float scaleX = displayWidth;
        float scaleY = displayHeight;

        if (videoAspect > displayAspect)
        {
            scaleY = displayWidth / videoAspect;
        }
        else
        {
            scaleX = displayHeight * videoAspect;
        }

        // Draw each keypoint
        for (int k = 0; k < 17; k++)
        {
            int keypointIndex = id * 17 + k;
            if (keypointIndex >= keypointPool.Count)
            {
                Debug.LogError($"Keypoint index {keypointIndex} out of range. Pool size: {keypointPool.Count}");
                continue;
            }

            float rawConfidence = box.keypoints[k * 3 + 2];
            float confidence = 1f / (1f + Mathf.Exp(-rawConfidence));

            float x = box.keypoints[k * 3] * scaleX - scaleX / 2;
            float y = -box.keypoints[k * 3 + 1] * scaleY + scaleY / 2;

            if (confidence > 0.2f)
            {
                GameObject keypoint = keypointPool[keypointIndex];
                keypoint.SetActive(true);
                keypoint.transform.localPosition = new Vector3(x, y, 0);
                RectTransform rt = keypoint.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(keypointSize, keypointSize);
            }
            else
            {
                keypointPool[keypointIndex].SetActive(false);
            }
        }
    }

    public void DrawSkeleton(BoundingBox box, int id)
    {
        int requiredLines = (id + 1) * skeletonConnections.Length;
        while (skeletonPool.Count < requiredLines)
        {
            var line = new GameObject("SkeletonLine");
            line.AddComponent<CanvasRenderer>();
            Image img = line.AddComponent<Image>();
            img.color = skeletonColor;
            line.transform.SetParent(displayLocation, false);
            skeletonPool.Add(line);
        }

        float displayWidth = displayImage.rectTransform.rect.width;
        float displayHeight = displayImage.rectTransform.rect.height;

        // Scale the coordinates to match the video aspect ratio
        float videoAspect = (float)video.width / video.height;
        float displayAspect = displayWidth / displayHeight;
        float scaleX = displayWidth;
        float scaleY = displayHeight;

        if (videoAspect > displayAspect)
        {
            scaleY = displayWidth / videoAspect;
        }
        else
        {
            scaleX = displayHeight * videoAspect;
        }

        // Draw each skeleton connection
        for (int s = 0; s < skeletonConnections.Length; s++)
        {
            int kp1 = skeletonConnections[s].Item1;
            int kp2 = skeletonConnections[s].Item2;

            float x1 = box.keypoints[kp1 * 3] * scaleX - scaleX / 2;
            float y1 = -box.keypoints[kp1 * 3 + 1] * scaleY + scaleY / 2;
            float conf1 = box.keypoints[kp1 * 3 + 2];

            float x2 = box.keypoints[kp2 * 3] * scaleX - scaleX / 2;
            float y2 = -box.keypoints[kp2 * 3 + 1] * scaleY + scaleY / 2;
            float conf2 = box.keypoints[kp2 * 3 + 2];

            if (conf1 > 0.2f && conf2 > 0.2f)
            {
                int lineIndex = id * skeletonConnections.Length + s;
                if (lineIndex >= skeletonPool.Count)
                {
                    Debug.LogError($"Skeleton line index {lineIndex} out of range. Pool size: {skeletonPool.Count}");
                    continue;
                }
                GameObject line = skeletonPool[lineIndex];
                line.SetActive(true);

                // Calculate line position and rotation
                Vector2 midPoint = new Vector2((x1 + x2) / 2, (y1 + y2) / 2);
                float angle = Mathf.Atan2(y2 - y1, x2 - x1) * Mathf.Rad2Deg;
                float length = Vector2.Distance(new Vector2(x1, y1), new Vector2(x2, y2));

                line.transform.localPosition = new Vector3(midPoint.x, midPoint.y, 0);
                line.transform.localRotation = Quaternion.Euler(0, 0, angle);

                RectTransform rt = line.GetComponent<RectTransform>();
                rt.sizeDelta = new Vector2(length, skeletonWidth);
            }
            else
            {
                skeletonPool[id * skeletonConnections.Length + s].SetActive(false);
            }
        }
    }

    public void ClearAnnotations()
    {
        foreach (var box in boxPool)
        {
            box.SetActive(false);
        }
        foreach (var keypoint in keypointPool)
        {
            keypoint.SetActive(false);
        }
        foreach (var skeleton in skeletonPool)
        {
            skeleton.SetActive(false);
        }
    }

    private void OnDestroy()
    {
        centersToCorners?.Dispose();
        worker?.Dispose();
    }

    Rect GetVideoRectInRawImage()
    {
        float rawWidth = displayImage.rectTransform.rect.width;
        float rawHeight = displayImage.rectTransform.rect.height;
        float videoAspect = (float)video.width / video.height;
        float rawAspect = rawWidth / rawHeight;

        float drawWidth, drawHeight, offsetX, offsetY;

        if (videoAspect > rawAspect)
        {
            // Video es más ancho, bandas arriba/abajo
            drawWidth = rawWidth;
            drawHeight = rawWidth / videoAspect;
            offsetX = 0;
            offsetY = (rawHeight - drawHeight) / 2;
        }
        else
        {
            // Video es más alto, bandas a los lados
            drawWidth = rawHeight * videoAspect;
            drawHeight = rawHeight;
            offsetX = (rawWidth - drawWidth) / 2;
            offsetY = 0;
        }
        return new Rect(offsetX, offsetY, drawWidth, drawHeight);
    }
}