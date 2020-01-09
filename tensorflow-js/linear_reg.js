"use strict"
// console.log('Hello World')

/**
 * Get car data reduced to just the variables we are intereseted 
 * and cleaned of missing data
*/

console.log(tf.getBackend());

let getData = (async()=>{
    const carDataReq = await fetch('https://storage.googleapis.com/tfjs-tutorials/carsData.json');
    const carData = await carDataReq.json();
    // console.log(carData);
    const cleaned = carData.map(car=>({
        mpg:car.Miles_per_Gallon,
        horespower:car.Horsepower
    }))
    .filter(car => (car.mpg!=null && car.horespower!=null));
    return cleaned;
})();

let run = (async()=>{
    const data = await getData;
    const values = data.map(d => ({
        x:d.horespower,
        y:d.mpg
    }));
    tfvis.render.scatterplot(
        {name:'Horespower v MPG'},
        {values},
        {
            xLabel:'Horespower',
            yLabel:'MPG',
            height:300
        }
    )
    // Creating the model
    const model = linearmodel;
    tfvis.show.modelSummary({name: 'Model Summary'}, model);
    const tensorData = convertToTensor(data);
    const {inputs, labels} = tensorData;
    await trainModel(model, inputs, labels);
    testModel(model,data,tensorData);
})();
document.addEventListener('DOMContentLoaded',run);

let linearmodel = (()=>{
    // Create a sequential model
    const model = tf.sequential();
    // Add hidden layer
    model.add(tf.layers.dense({inputShape:[1], units:1, useBias:true}));
    model.add(tf.layers.dense({units:50, activation:'sigmoid', useBias:true}));
    // Adding the output layer
    model.add(tf.layers.dense({units:1, useBias:true}));
    return model;
})();



/**
 * Convert the input data to tensors that we can use for machine
 * learning. We will also do the important best practices of _shuffling_
 * the data and _normalizing_ the data
 * MPG on the y-axis
 */
let convertToTensor = data =>{
    return tf.tidy(()=>{
        tf.util.shuffle(data);

        const inputs = data.map(d=>d.horespower);
        const labels = data.map(d=>d.mpg);

        const inputTensor = tf.tensor2d(inputs, [inputs.length,1]);
        const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

        const inputMax = inputTensor.max();
        const inputMin = inputTensor.min();
        const labelMax = labelTensor.max();
        const labelMin = labelTensor.min();

        const normalizedInputs = inputTensor.sub(inputMin).div(inputMax.sub(inputMin));
        const normalizedLabels = labelTensor.sub(labelMin).div(labelMax.sub(labelMin));
        return {
            inputs : normalizedInputs,
            labels : normalizedLabels,
            inputMax,
            inputMin,
            labelMax,
            labelMin,
        }
    });
};

let trainModel = async(model, inputs, labels)=>{
    model.compile({
        optimizer:tf.train.adam(),
        loss:tf.losses.meanSquaredError,
        metrics: ['mse'],
    });
    const batchsize = 32;
    const epochs = 100;

    return await model.fit(inputs, labels, {
        batchsize,
        epochs,
        shuffle:true,
        callbacks:tfvis.show.fitCallbacks(
            {name : 'Training performance',},
            ['loss','mse'],
            {height:200, callbacks:['onEpochEnd']}
            )
    });

};


let testModel = (model, inputData, normalizationData)=>{
    const {inputMax, inputMin, labelMin, labelMax} = normalizationData
    const [xs, preds] = tf.tidy(()=>{
        const xs = tf.linspace(0,1,100);
        const preds = model.predict(xs.reshape([100,1]));

        const uNormXs = xs.mul(inputMax.sub(inputMin)).add(inputMin);
        const uNormPred = preds.mul(labelMax.sub(labelMin)).add(labelMin);

        return [uNormXs.dataSync(), uNormPred.dataSync()];
    });
    const predictedPoints = Array.from(xs).map((val,i) =>({x:val, y:preds[i]}));
    const originalPoints = inputData.map(d =>({
        x:d.horespower, y:d.mpg
    }));
    console.log(originalPoints)

    tfvis.render.scatterplot(
        {name : 'Model Predictions vs Original data'},
        {values : [originalPoints, predictedPoints], series:['original','predicted']},
        {
            xLabel:'Horsepower',
            yLabel:'MPG',
            height:300
        }
    );

};
