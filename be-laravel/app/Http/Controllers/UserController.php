<?php

namespace App\Http\Controllers;

use App\Models\User;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class UserController extends Controller
{
    //
    public function createUser()
    {
        $user = new User;
        $user->name = 'Raey Faldo';
        $user->email = 'raey3221@gmail.com';
        $user->password = Hash::make("mantap123");
        return $user->save();
    }

    public function createToken(Request $request)
    {
        
    }
}
