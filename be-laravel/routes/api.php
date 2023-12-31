<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\UserController;
use App\Http\Controllers\PredictionController;

/*
|--------------------------------------------------------------------------
| API Routes
|--------------------------------------------------------------------------
|
| Here is where you can register API routes for your application. These
| routes are loaded by the RouteServiceProvider and all of them will
| be assigned to the "api" middleware group. Make something great!
|
*/

Route::middleware('auth:sanctum')->get('/user', function (Request $request) {
    return $request->user();
});

Route::get('/create-user', [UserController::class, 'createUser']);
Route::get('/get-token', [UserController::class, 'createToken']);
Route::get('/get-latest-prediction/{location_id}', [PredictionController::class, 'getLatestPrediction']);
Route::get('/get-24-prediction/{location_id}', [PredictionController::class, 'get24Prediction']);
Route::post('/store-prediction', [PredictionController::class, 'storePrediction']);
