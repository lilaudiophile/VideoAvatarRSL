const tf = require('@tensorflow/tfjs-node');
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');
const ffmpegPath = '/opt/homebrew/bin/ffmpeg';
const ffmpeg = require('fluent-ffmpeg');
const async = require('async');

ffmpeg.setFfmpegPath(ffmpegPath);
const uniqueClasses = new Set(); // Для хранения уникальных классов

const allFrames = []; // Массив для всех кадров
const labelsArray = []; // Массив для всех меток
const batchSize = 32; // Размер батча

// Функция для извлечения кадров и сохранения их в тензоры
async function extractFramesFromVideo(videoPath, frameRate, height, width) {
    console.log(`Начинается извлечение кадров из: ${videoPath}`);

    const framesDir = path.join(__dirname, 'frames');
    if (!fs.existsSync(framesDir)) {
        fs.mkdirSync(framesDir);
    }

    return new Promise((resolve, reject) => {
        ffmpeg(videoPath)
            .on('error', (err) => {
                console.error(`Ошибка извлечения кадров: ${err}`);
                reject(err);
            })
            .outputOptions('-vf', `fps=${frameRate}`)
            .outputOptions('-s', `${width}x${height}`)
            .on('end', async () => {
                console.log(`Кадры успешно извлечены из ${videoPath}`);

                const frameFiles = fs.readdirSync(framesDir);
                const frames = frameFiles.map(file => {
                    const imageData = fs.readFileSync(path.join(framesDir, file));
                    let frameTensor = tf.node.decodeImage(imageData, 3); // Декодируем изображение в тензор
                    frameTensor = tf.image.resizeBilinear(frameTensor, [480, 640]); // Изменяем размер кадра
                    return frameTensor;
                });

                resolve(frames); // Возвращаем массив тензоров
            })
            .save(path.join(framesDir, 'temp_frame_%04d.jpg'))
            .run();
    });
}


// Очередь для управления параллельной обработкой видео
const queue = async.queue(async (data, callback) => {
    console.log(`Обрабатывается видео: ${data.attachment_id}`);
    try {
        const videoFolder = data.train === 'True' ? 'train' : 'test';
        const videoFileName = `${data.attachment_id}.mp4`;
        const videoPath = path.join(__dirname, '..', 'archive', 'slovo', videoFolder, videoFileName);

        const label = data.text && typeof data.text === 'string' ? data.text.trim() : '';
        if (label) {
            console.log(`Добавляется в Set: "${label}"`);
            uniqueClasses.add(label);
            labelsArray.push(label); // Добавляем метку для каждого видео
        } else {
            console.error(`Предупреждение: пустая метка для видео ${data.attachment_id}`);
        }

        const height = parseInt(data.height, 10);
        const width = parseInt(data.width, 10);
        if (isNaN(height) || isNaN(width)) {
            console.error(`Ошибка: неправильные размеры для видео ${data.attachment_id}`);
            callback();
            return;
        }

        const frameRate = 3; // Частота кадров
        const frames = await extractFramesFromVideo(videoPath, frameRate, height, width);
        if (frames.length === 0) {
            console.error(`Ошибка: не извлечено кадров для видео ${data.attachment_id}`);
            callback(new Error(`Не извлечены кадры для видео: ${data.attachment_id}`));
            return;
        }

        allFrames.push(...frames); // Сохраняем все кадры в общем массиве
        console.log(`Обработано видео: ${data.attachment_id}, метка: "${label}"`);
        callback();
    } catch (error) {
        console.error(`Ошибка обработки видео для ${data.attachment_id}: ${error}`);
        callback(error);
    }
}, 3); // Одновременно обрабатываем не более 3 видеофайлов

// Чтение файла с аннотациями
fs.createReadStream('archive/slovo/annotations.csv', { encoding: 'utf8' })
    .pipe(csv({ separator: '\t' }))
    .on('data', (data) => {
        console.log(`Читается строка: ${JSON.stringify(data)}`);
        queue.push(data);
    })
    .on('end', () => {
        console.log('Все видео добавлены в очередь на обработку.');
        checkQueue();
    })
    .on('error', (error) => {
        console.error(`Ошибка чтения CSV файла: ${error.message}`);
    });

// Функция для проверки состояния очереди
function checkQueue() {
    if (queue.length() === 0 && queue.running() === 0) {
        console.log('Обработка всех видео завершена.');
        processVideosAndTrainModel(); // Начинаем обучение модели после завершения обработки всех данных
    } else {
        setImmediate(checkQueue);
    }
}

// Подготовка меток в виде one-hot кодировок
function prepareLabels(uniqueClasses, labels) {
    const classArray = Array.from(uniqueClasses);
    return labels.map(label => {
        const index = classArray.indexOf(label);
        return tf.oneHot(index, uniqueClasses.size);
    });
}

// Функция для обучения модели на батче данных
async function trainOnBatch(framesBatch, labelsBatch, model) {
    const inputTensor = tf.stack(framesBatch);
    const labelsTensor = tf.stack(prepareLabels(uniqueClasses, labelsBatch));

    const [trainXs, valXs] = tf.split(inputTensor, 2);
    const [trainYs, valYs] = tf.split(labelsTensor, 2);

    const history = await model.fit(trainXs, trainYs, {
        epochs: 10,
        validationData: [valXs, valYs],
        batchSize: batchSize,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Эпоха ${epoch + 1} завершена. Точность: ${logs.acc}, Потери: ${logs.loss}`);
            }
        }
    });

    // Освобождение памяти
    tf.dispose([inputTensor, labelsTensor, trainXs, trainYs, valXs, valYs]);
}

// Основная функция для обработки видео и обучения модели
async function processVideosAndTrainModel() {
    const frameHeight = 480;
    const frameWidth = 640;

    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [frameHeight, frameWidth, 3],
        filters: 32,
        kernelSize: 3,
        activation: 'relu',
    }));
    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));

    model.add(tf.layers.flatten()); // Превращаем тензор в одномерный для подачи в LSTM
    model.add(tf.layers.dense({ units: 128, activation: 'relu' }));
    model.add(tf.layers.dense({ units: uniqueClasses.size, activation: 'softmax' }));

    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    let framesBatch = [];
    let labelsBatch = [];

    for (let i = 0; i < allFrames.length; i++) {
        framesBatch.push(allFrames[i]);
        labelsBatch.push(labelsArray[i]);

        if (framesBatch.length === batchSize || i === allFrames.length - 1) {
            // Если набрали полный батч или это последний элемент, запускаем обучение
            await trainOnBatch(framesBatch, labelsBatch, model);
            framesBatch = []; // Очищаем батч для следующей партии
            labelsBatch = [];
        }
    }

    console.log('Обучение завершено.');
}
