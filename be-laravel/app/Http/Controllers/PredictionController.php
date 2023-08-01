<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\Prediction;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\DB;

class PredictionController extends Controller
{
    //
    public function storePrediction(Request $request)
    {
        DB::beginTransaction();
        try {
            $prediction = new Prediction;
            $prediction->pm10 = $request->pm10;
            $prediction->pm25 = $request->pm25;
            $prediction->so2 = $request->so2;
            $prediction->co = $request->co;
            $prediction->o3 = $request->o3;
            $prediction->no2 = $request->no2;
            $prediction->prediction_result = $request->prediction_result;
            $prediction->accuracy = $request->accuracy;

            if ($prediction->save()) {
                DB::commit();
                return json_encode([
                    "status" => true,
                    "message" => "Success"
                ]);
            } else {
                DB::rollBack();
                return json_encode([
                    "status" => false,
                    "message" => "Error"
                ]);
            }
        } catch (\Throwable $th) {
            DB::rollBack();
            return json_encode([
                "status" => false,
                "message" => "Error",
                "data" => $th
            ]);
        }
    }
}
