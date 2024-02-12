<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\Prediction;
use Illuminate\Http\Request;
use App\Events\NewDataInserted;
use Illuminate\Support\Facades\DB;

class PredictionController extends Controller
{
    //
    public function storePrediction(Request $request)
    {
        DB::beginTransaction();
        try {
            $latest_data = Prediction::where('location', $request->location)->latest()->first();
            if (
                $latest_data &&
                $latest_data->pm10 == $request->pm10 &&
                $latest_data->pm25 == $request->pm25 &&
                $latest_data->so2 == $request->so2 &&
                $latest_data->co == $request->co &&
                $latest_data->o3 == $request->o3 &&
                $latest_data->no2 == $request->no2
            ) {
                return response()->json([
                    "status" => false,
                    "message" => "Data Tidak Berubah"
                ]);
            }

            $prediction = new Prediction;
            $prediction->pm10 = $request->pm10;
            $prediction->pm25 = $request->pm25;
            $prediction->so2 = $request->so2;
            $prediction->co = $request->co;
            $prediction->o3 = $request->o3;
            $prediction->no2 = $request->no2;
            $prediction->prediction_result = $request->prediction_result;
            $prediction->accuracy = $request->accuracy;
            $prediction->location = $request->location;

            if ($prediction->save()) {
                DB::commit();
                event(new NewDataInserted($prediction->toArray()));
                return response()->json([
                    "status" => true,
                    "message" => "Success"
                ]);
            } else {
                DB::rollBack();
                return response()->json([
                    "status" => false,
                    "message" => "Error"
                ]);
            }
        } catch (\Throwable $th) {
            DB::rollBack();
            return response()->json([
                "status" => false,
                "message" => "Error",
                "data" => $th
            ]);
        }
    }

    public function getLatestPrediction($location_id)
    {
        $data = Prediction::where('location', $location_id)->latest()->first();

        return [
            "status" => true,
            "message" => "Success",
            "data" => $data ? $data->toArray() : []
        ];
    }

    public function get24Prediction($location_id)
    {
        $data = Prediction::query()
            ->where('location', $location_id)
            ->orderBy('id', 'desc')
            ->limit(24)
            ->get();

        return [
            "status" => true,
            "message" => "Success",
            "data" => $data ? $data->toArray() : []
        ];
    }
}
